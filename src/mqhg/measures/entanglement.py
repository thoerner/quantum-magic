"""Entanglement measures: von Neumann entropy, Rényi entropy, mutual information."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector


def von_neumann_entropy(rho: NDArray[np.complex128]) -> float:
    """S(ρ) = -Tr(ρ log₂ ρ) via eigenvalue decomposition."""
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(-np.sum(eigenvalues * np.log2(eigenvalues)))


def renyi_entropy(rho: NDArray[np.complex128], alpha: float = 2.0) -> float:
    """S_α(ρ) = (1/(1-α)) log₂ Tr(ρ^α).

    alpha=1 limit gives von Neumann. alpha=2 gives collision entropy.
    """
    if np.isclose(alpha, 1.0):
        return von_neumann_entropy(rho)
    eigenvalues = np.linalg.eigvalsh(rho)
    eigenvalues = eigenvalues[eigenvalues > 1e-15]
    return float(np.log2(np.sum(eigenvalues**alpha)) / (1 - alpha))


def subsystem_entropy(state: Statevector, subsystem: list[int], alpha: float = 1.0) -> float:
    """Compute entropy of a subsystem (von Neumann or Rényi)."""
    rho = state.reduced_density_matrix(subsystem)
    if np.isclose(alpha, 1.0):
        return von_neumann_entropy(rho)
    return renyi_entropy(rho, alpha)


def mutual_information(state: Statevector, A: list[int], B: list[int]) -> float:
    """I(A:B) = S(A) + S(B) - S(A∪B)."""
    s_a = subsystem_entropy(state, A)
    s_b = subsystem_entropy(state, B)
    s_ab = subsystem_entropy(state, sorted(set(A) | set(B)))
    return s_a + s_b - s_ab


def mutual_information_matrix(state: Statevector) -> NDArray[np.float64]:
    """Compute pairwise mutual information I(i:j) for all qubit pairs."""
    n = state.n_qubits
    mi = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            mi_ij = mutual_information(state, [i], [j])
            mi[i, j] = mi_ij
            mi[j, i] = mi_ij
    return mi


def entanglement_spectrum(state: Statevector, subsystem: list[int]) -> NDArray[np.float64]:
    """Eigenvalues of the reduced density matrix (entanglement spectrum)."""
    rho = state.reduced_density_matrix(subsystem)
    eigenvalues = np.linalg.eigvalsh(rho)
    return np.sort(eigenvalues)[::-1]


def bipartite_entropy_profile(state: Statevector) -> list[float]:
    """Entropy S(A) for A = {0}, {0,1}, {0,1,2}, ..., {0,...,n/2-1}."""
    n = state.n_qubits
    profile = []
    for k in range(1, n // 2 + 1):
        profile.append(subsystem_entropy(state, list(range(k))))
    return profile
