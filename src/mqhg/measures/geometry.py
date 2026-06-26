"""Emergent geometry extraction from quantum states.

Defines geometry proxies based on mutual information, entanglement structure,
graph curvature, and spectral properties.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import networkx as nx

from ..core.statevector import Statevector
from ..core.gates import pauli_tensor
from .entanglement import mutual_information_matrix


def mutual_info_distance_matrix(
    state: Statevector,
    epsilon: float = 1e-10,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Emergent distance from mutual information.

    d(i,j) = max(0, -ξ * log(I(i:j) + ε))

    High mutual information → short distance.
    Low mutual information → long distance.
    Clamped to non-negative to ensure valid metric for Dijkstra.
    """
    mi = mutual_information_matrix(state)
    with np.errstate(divide="ignore"):
        distances = -scale * np.log(mi + epsilon)
    np.maximum(distances, 0.0, out=distances)
    np.fill_diagonal(distances, 0.0)
    return distances


def emergent_metric(state: Statevector, epsilon: float = 1e-10) -> NDArray[np.float64]:
    """Graph-geodesic metric from mutual-information distances.

    Computes shortest paths on the MI-distance weighted complete graph.
    This ensures the triangle inequality holds.
    """
    raw_dist = mutual_info_distance_matrix(state, epsilon=epsilon)
    n = state.n_qubits

    G = nx.complete_graph(n)
    for i in range(n):
        for j in range(i + 1, n):
            G[i][j]["weight"] = raw_dist[i, j]

    # Shortest-path metric
    metric = np.zeros((n, n), dtype=np.float64)
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    for i in range(n):
        for j in range(n):
            metric[i, j] = lengths[i][j]

    return metric


def correlator_distance_matrix(
    state: Statevector,
    epsilon: float = 1e-10,
    scale: float = 1.0,
) -> NDArray[np.float64]:
    """Emergent distance from connected Pauli-Pauli correlators.

    d(i,j) = -ξ * log(C(i,j) + ε)

    where C(i,j) = Σ_{P,Q ∈ {X,Y,Z}} |⟨P_i Q_j⟩ - ⟨P_i⟩⟨Q_j⟩|²

    More sensitive than MI for detecting geometry changes from local
    excitations, especially in highly entangled states where pairwise
    single-qubit MI is degenerate.
    """
    n = state.n_qubits
    paulis = ["X", "Y", "Z"]

    single_exp: dict[tuple[int, str], complex] = {}
    for i in range(n):
        for p in paulis:
            P = pauli_tensor(p)
            single_exp[(i, p)] = state.expectation(P, [i])

    distances = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            corr_sum = 0.0
            for p in paulis:
                for q in paulis:
                    PQ = np.kron(pauli_tensor(p), pauli_tensor(q))
                    exp_pq = state.expectation(PQ, [i, j])
                    connected = exp_pq - single_exp[(i, p)] * single_exp[(j, q)]
                    corr_sum += abs(connected) ** 2
            d = -scale * np.log(np.sqrt(corr_sum) + epsilon)
            distances[i, j] = d
            distances[j, i] = d

    return distances


def connected_correlator_matrix(state: Statevector) -> NDArray[np.float64]:
    """Connected two-point correlation strength matrix G.

    G[i,j] = sqrt( Σ_{P,Q ∈ {X,Y,Z}} |⟨P_i Q_j⟩ - ⟨P_i⟩⟨Q_j⟩|² )

    Involves only weight-≤2 Pauli observables, so it is efficiently estimable
    from classical shadows (see hardware.fast_shadows). ||G||_F is used as a
    hardware-measurable backreaction proxy in the scaling study.
    """
    n = state.n_qubits
    paulis = ["X", "Y", "Z"]

    single_exp: dict[tuple[int, str], complex] = {}
    for i in range(n):
        for p in paulis:
            single_exp[(i, p)] = state.expectation(pauli_tensor(p), [i])

    g = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            corr_sum = 0.0
            for p in paulis:
                for q in paulis:
                    PQ = np.kron(pauli_tensor(p), pauli_tensor(q))
                    connected = state.expectation(PQ, [i, j]) - single_exp[(i, p)] * single_exp[(j, q)]
                    corr_sum += abs(connected) ** 2
            g[i, j] = g[j, i] = np.sqrt(corr_sum)
    return g


def mutual_info_graph(
    state: Statevector,
    threshold: float = 0.01,
) -> nx.Graph:
    """Construct a weighted graph where edge weights are mutual information values.

    Only includes edges with I(i:j) > threshold.
    """
    mi = mutual_information_matrix(state)
    n = state.n_qubits

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if mi[i, j] > threshold:
                G.add_edge(i, j, weight=mi[i, j])

    return G


def ollivier_ricci_curvature(
    state: Statevector,
    i: int,
    j: int,
    epsilon: float = 1e-10,
) -> float:
    """Ollivier-Ricci curvature proxy for edge (i,j) in the MI graph.

    κ(i,j) = 1 - W₁(μᵢ, μⱼ) / d(i,j)

    where μᵢ is the probability distribution on neighbors of i weighted by MI,
    and W₁ is the Wasserstein-1 distance using the emergent metric.

    Positive curvature → sphere-like (converging geodesics)
    Zero curvature → flat
    Negative curvature → hyperbolic (diverging geodesics)
    """
    mi = mutual_information_matrix(state)
    metric = emergent_metric(state, epsilon=epsilon)
    n = state.n_qubits

    def neighbor_distribution(node: int) -> NDArray[np.float64]:
        weights = mi[node].copy()
        weights[node] = 0.0
        total = weights.sum()
        if total < 1e-15:
            # Uniform if no correlations
            dist = np.ones(n) / (n - 1)
            dist[node] = 0.0
            return dist
        return weights / total

    mu_i = neighbor_distribution(i)
    mu_j = neighbor_distribution(j)

    # Wasserstein-1 via linear programming (exact for small n)
    from scipy.optimize import linprog

    # Cost matrix flattened
    cost = metric.flatten()
    n_vars = n * n

    # Constraints: row sums = mu_i, col sums = mu_j
    A_eq = np.zeros((2 * n, n_vars), dtype=np.float64)
    b_eq = np.zeros(2 * n, dtype=np.float64)

    for k in range(n):
        # Row sum constraint
        A_eq[k, k * n:(k + 1) * n] = 1.0
        b_eq[k] = mu_i[k]
        # Column sum constraint
        A_eq[n + k, k::n] = 1.0
        b_eq[n + k] = mu_j[k]

    bounds = [(0, None)] * n_vars
    result = linprog(cost, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

    if result.success:
        w1 = result.fun
    else:
        w1 = 0.0

    d_ij = metric[i, j]
    if d_ij < 1e-15:
        return 0.0

    return 1.0 - w1 / d_ij


def average_curvature(state: Statevector, threshold: float = 0.01) -> float:
    """Average Ollivier-Ricci curvature over significant edges."""
    mi = mutual_information_matrix(state)
    n = state.n_qubits

    curvatures = []
    for i in range(n):
        for j in range(i + 1, n):
            if mi[i, j] > threshold:
                kappa = ollivier_ricci_curvature(state, i, j)
                curvatures.append(kappa)

    if not curvatures:
        return 0.0
    return float(np.mean(curvatures))


def spectral_dimension(state: Statevector, epsilon: float = 1e-10) -> float:
    """Spectral dimension from the graph Laplacian of the MI graph.

    Estimated from the scaling of the heat kernel trace.
    For small systems, returns the effective dimension from eigenvalue spacing.
    """
    mi = mutual_information_matrix(state)
    n = state.n_qubits

    # Construct weighted Laplacian
    degree = np.sum(mi, axis=1)
    laplacian = np.diag(degree) - mi

    eigenvalues = np.sort(np.linalg.eigvalsh(laplacian))
    # Skip zero eigenvalue
    nonzero_eigs = eigenvalues[eigenvalues > 1e-10]

    if len(nonzero_eigs) < 2:
        return 0.0

    # Spectral dimension from return probability scaling:
    # d_s ≈ -2 * d(log P(t)) / d(log t) at intermediate t
    # Approximate via eigenvalue distribution
    log_eigs = np.log(nonzero_eigs)
    log_counts = np.log(np.arange(1, len(nonzero_eigs) + 1))

    # Linear fit gives effective dimension
    if len(log_eigs) >= 2:
        slope = np.polyfit(log_eigs, log_counts, 1)[0]
        return float(max(0.0, 2 * slope))

    return 1.0
