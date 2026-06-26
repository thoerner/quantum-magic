"""SRE estimation from hardware shadow data.

Estimates stabilizer Rényi entropy without enumerating all 4^n Pauli strings.
Uses importance sampling: randomly sample Pauli strings, estimate |<P>|² for
each via shadow tomography, and compute the SRE from the sample.

This trades exponential exact computation for polynomial statistical estimation,
enabling SRE measurement on quantum hardware beyond n=14 qubits.
"""

from __future__ import annotations

from itertools import product as iproduct

import numpy as np
from numpy.typing import NDArray

from .shadows import ShadowData, ShadowProtocol


def estimate_sre_from_shadows(
    shadow_data: ShadowData,
    n_pauli_samples: int = 500,
    alpha: int = 2,
    seed: int | None = None,
) -> float:
    """Estimate SRE from shadow tomography data via importance sampling.

    Instead of summing over all 4^n Pauli strings, we:
    1. Sample random Pauli strings uniformly from {I,X,Y,Z}^n
    2. Estimate |<P>|² for each using the shadow data
    3. Compute SRE from the sample average

    The estimator for M_2 is:
        M_2 ≈ -log₂[ (4^n / 2^n) * (1/K) Σ_{k=1}^K |<P_k>|⁴ ] - n
            = -log₂[ 2^n * mean(|<P_k>|⁴) ] - n

    where P_k are uniformly sampled Pauli strings (including identity).

    Args:
        shadow_data: Classical shadow data from hardware.
        n_pauli_samples: Number of random Pauli strings to sample.
        alpha: Rényi index (default 2).
        seed: Random seed for Pauli sampling.

    Returns:
        Estimated SRE value.
    """
    rng = np.random.default_rng(seed)
    n = shadow_data.n_qubits
    num_paulis = 4**n

    # Sample random Pauli strings
    pauli_chars = "IXYZ"
    xi_values = []

    for _ in range(n_pauli_samples):
        ps = "".join(rng.choice(list(pauli_chars)) for _ in range(n))
        exp_val = ShadowProtocol.estimate_pauli_expectation(shadow_data, ps)
        xi = exp_val**2  # |<P>|²
        xi_values.append(xi)

    xi_arr = np.array(xi_values, dtype=np.float64)

    if alpha == 2:
        # M_2 = -log₂[ Σ_P |<P>|⁴ / 4^n ] - n
        # With uniform sampling: Σ_P f(P) / 4^n ≈ mean(f(P_k))
        mean_xi_sq = np.mean(xi_arr**2)
        if mean_xi_sq < 1e-30:
            return float(n)
        m2 = -np.log2(mean_xi_sq) - n
        return float(max(0.0, m2))
    else:
        mean_xi_alpha = np.mean(xi_arr**alpha)
        if mean_xi_alpha < 1e-30:
            return float(n)
        m_alpha = (1 / (1 - alpha)) * np.log2(mean_xi_alpha) - n
        return float(max(0.0, m_alpha))


def estimate_nonlocal_magic_from_shadows(
    shadow_data: ShadowData,
    partition: list[list[int]] | None = None,
    n_pauli_samples: int = 500,
    seed: int | None = None,
) -> float:
    """Estimate non-local magic from shadow data.

    Non-local magic = total SRE - sum of local (subsystem) SREs.

    For the local SRE of subsystem A, we only sample Pauli strings
    that act as identity on the complement of A.

    Args:
        shadow_data: Classical shadow data from hardware.
        partition: Subsystem partition. Defaults to single-qubit partition.
        n_pauli_samples: Pauli samples per estimation.
        seed: Random seed.

    Returns:
        Estimated non-local magic.
    """
    rng = np.random.default_rng(seed)
    n = shadow_data.n_qubits

    if partition is None:
        partition = [[i] for i in range(n)]

    # Total SRE
    total_sre = estimate_sre_from_shadows(shadow_data, n_pauli_samples, seed=rng.integers(2**31))

    # Local SRE for each subsystem
    local_sre_sum = 0.0
    pauli_chars = "IXYZ"

    for subsys in partition:
        k = len(subsys)
        local_xi_values = []

        for _ in range(n_pauli_samples):
            # Build Pauli string: non-trivial only on subsys qubits
            ps = ["I"] * n
            for q in subsys:
                ps[q] = rng.choice(list(pauli_chars))

            ps_str = "".join(ps)
            exp_val = ShadowProtocol.estimate_pauli_expectation(shadow_data, ps_str)
            xi = exp_val**2
            local_xi_values.append(xi)

        local_arr = np.array(local_xi_values, dtype=np.float64)
        mean_xi_sq = np.mean(local_arr**2)
        if mean_xi_sq < 1e-30:
            local_sre_sum += float(k)
        else:
            m2_local = -np.log2(mean_xi_sq) - k
            local_sre_sum += max(0.0, m2_local)

    return max(0.0, total_sre - local_sre_sum)


def confidence_interval(
    shadow_data: ShadowData,
    n_pauli_samples: int = 500,
    n_bootstrap: int = 100,
    alpha: int = 2,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for SRE estimate.

    Returns:
        Tuple of (mean, lower_95, upper_95).
    """
    rng = np.random.default_rng(seed)
    estimates = []

    for _ in range(n_bootstrap):
        # Resample shadows with replacement
        n_shadows = shadow_data.n_snapshots
        indices = rng.choice(n_shadows, size=n_shadows, replace=True)
        resampled = ShadowData(n_qubits=shadow_data.n_qubits)
        for idx in indices:
            s = shadow_data.shadows[idx]
            resampled.add(s.basis, s.outcome)

        est = estimate_sre_from_shadows(resampled, n_pauli_samples, alpha, seed=rng.integers(2**31))
        estimates.append(est)

    estimates_arr = np.array(estimates)
    mean = float(np.mean(estimates_arr))
    lower = float(np.percentile(estimates_arr, 2.5))
    upper = float(np.percentile(estimates_arr, 97.5))
    return mean, lower, upper


def required_measurements(
    n_qubits: int,
    target_precision: float = 0.1,
    confidence: float = 0.95,
    n_observables: int = 500,
) -> int:
    """Estimate the number of shadow measurements needed.

    Based on the shadow tomography bound:
    N = O(max_weight * log(M) / ε²)

    where max_weight is the maximum Pauli weight and M is the number
    of observables to estimate.

    Args:
        n_qubits: System size.
        target_precision: Desired precision ε on each <P> estimate.
        confidence: Confidence level (used for log factor).
        n_observables: Number of Pauli expectations to estimate.

    Returns:
        Recommended number of measurement circuits.
    """
    log_factor = np.log(2 * n_observables / (1 - confidence))
    # For random Pauli measurements, the overhead is 3^weight
    # Average weight for random Pauli strings on n qubits is 3n/4
    avg_weight = 3 * n_qubits / 4
    overhead = 3**avg_weight

    # Practical estimate (less conservative than worst-case bound)
    n_measurements = int(np.ceil(34 * log_factor / target_precision**2))

    # Cap at a practical maximum
    return min(n_measurements, 100_000)
