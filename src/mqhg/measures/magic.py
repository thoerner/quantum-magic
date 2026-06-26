"""Magic (non-stabilizerness) measures.

Implements:
- Stabilizer Rényi entropy (SRE): measures deviation from stabilizer states
  via Pauli expectation values.
- Mana: logarithmic negativity of the discrete Wigner function.
- Non-local magic: magic not attributable to individual subsystems.

Performance: groups Pauli strings by flip_mask and uses Fast Walsh-Hadamard
Transform to evaluate all phase-mask inner products simultaneously.
Total SRE cost: O(n * 4^n) instead of O(16^n) for dense matrices.
SRE-only calls use O(2^n) memory; full distribution calls use O(4^n).
"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector
from ..core.gates import pauli_tensor


# --- Fast Walsh-Hadamard Transform ---


def _fwht_inplace(a: NDArray) -> None:
    """In-place Fast Walsh-Hadamard Transform. Length must be a power of 2.

    Computes: a[k] <- sum_j a[j] * (-1)^popcount(k & j)
    """
    n = len(a)
    h = 1
    while h < n:
        blocks = a.reshape(-1, 2 * h)
        lo = blocks[:, :h].copy()
        hi = blocks[:, h:].copy()
        blocks[:, :h] = lo + hi
        blocks[:, h:] = lo - hi
        h <<= 1


# --- Index mapping ---


def _fm_pm_to_pauli_id(fm: int, pms: NDArray, n: int) -> NDArray:
    """Map (flip_mask, phase_mask) pairs to base-4 Pauli string IDs.

    I(0,0)->0, X(1,0)->1, Y(1,1)->2, Z(0,1)->3.
    """
    result = np.zeros(len(pms), dtype=np.int64)
    pow4 = np.int64(1)
    for k in range(n):
        fm_k = np.int64((fm >> k) & 1)
        pm_k = (pms >> k) & 1
        q_k = fm_k + pm_k * (3 - 2 * fm_k)
        result += q_k * pow4
        pow4 *= 4
    return result


# --- Core computation ---


def _fast_sre_direct(psi: NDArray, n: int, alpha: int = 2) -> float:
    """Compute SRE without storing full 4^n distribution. O(2^n) memory.

    Groups Paulis by flip_mask (2^n groups), applies FWHT within
    each group, and accumulates the SRE sum incrementally.
    """
    dim = 1 << n
    num_paulis = 4**n
    indices = np.arange(dim, dtype=np.int64)
    accumulator = 0.0

    for fm in range(dim):
        base = np.conj(psi[indices ^ fm]) * psi
        _fwht_inplace(base)
        xi = np.real(base * np.conj(base))
        if alpha == 2:
            accumulator += np.sum(xi * xi)
        else:
            accumulator += np.sum(xi**alpha)

    if accumulator < 1e-30:
        return float(n)
    if alpha == 2:
        return float(max(0.0, -np.log2(accumulator / num_paulis) - n))
    else:
        return float(
            max(0.0, (1 / (1 - alpha)) * np.log2(accumulator / num_paulis) - n)
        )


def _fast_pauli_expectations_sq(psi: NDArray, n: int) -> NDArray:
    """Compute |<psi|P|psi>|^2 for all 4^n Pauli strings.

    Returns full distribution. Uses O(4^n) memory — feasible up to n~14.
    For SRE-only, prefer _fast_sre_direct which uses O(2^n) memory.
    """
    dim = 1 << n
    num_paulis = 4**n
    indices = np.arange(dim, dtype=np.int64)
    pm_range = np.arange(dim, dtype=np.int64)
    results = np.zeros(num_paulis, dtype=np.float64)

    for fm in range(dim):
        base = np.conj(psi[indices ^ fm]) * psi
        _fwht_inplace(base)
        result_idx = _fm_pm_to_pauli_id(fm, pm_range, n)
        results[result_idx] = np.real(base * np.conj(base))

    return results


# --- Public API ---


def stabilizer_renyi_entropy(state: Statevector, alpha: int = 2) -> float:
    """Stabilizer Rényi entropy M_α(|ψ>).

    M_2 = -log₂[ (1/2^n) Σ_P <ψ|P|ψ>^4 ] - n

    Zero for stabilizer states, positive for magic states.
    Cost: O(n * 4^n) via FWHT, using only O(2^n) memory.
    """
    return _fast_sre_direct(state.amplitudes, state.n_qubits, alpha)


def pauli_expectation_distribution(state: Statevector) -> NDArray[np.float64]:
    """Compute |<ψ|P|ψ>|² for all Pauli strings P.

    Returns array of 4^n values. Uses O(4^n) memory.
    """
    return _fast_pauli_expectations_sq(state.amplitudes, state.n_qubits)


def mana(state: Statevector) -> float:
    """Mana (sum-negativity of discrete Wigner function).

    Pauli-based approximation: mana ≈ log₂(Σ_P |<P>|) - n/2.
    O(2^n) memory via incremental accumulation.
    """
    n = state.n_qubits
    dim = 1 << n
    indices = np.arange(dim, dtype=np.int64)
    psi = state.amplitudes
    sum_abs = 0.0

    for fm in range(dim):
        base = np.conj(psi[indices ^ fm]) * psi
        _fwht_inplace(base)
        sum_abs += np.sum(np.abs(base))

    return float(max(0.0, np.log2(sum_abs) - n))


def nonlocal_magic(
    state: Statevector,
    partition: list[list[int]] | None = None,
) -> float:
    """Non-local magic: total magic minus sum of local (subsystem) magic.

    M_nonlocal = M_total - Σ_i M(ρ_i)

    If partition is None, uses single-qubit partition.
    Non-local magic is the quantity proposed to correlate with
    gravitational backreaction.
    """
    n = state.n_qubits
    if partition is None:
        partition = [[i] for i in range(n)]

    total_magic = _fast_sre_direct(state.amplitudes, n, alpha=2)

    local_magic = 0.0
    for subsys in partition:
        local_magic += _subsystem_sre(state, subsys)

    return max(0.0, total_magic - local_magic)


# --- Internal helpers ---


def _all_pauli_strings(n: int) -> list[str]:
    """Generate all 4^n Pauli strings on n qubits (including identity)."""
    return ["".join(s) for s in product("IXYZ", repeat=n)]


def _sre_from_distribution(
    expectations_sq: NDArray, num_paulis: int, n: int, alpha: int = 2
) -> float:
    """Compute SRE from precomputed |<P>|^2 array."""
    if alpha == 2:
        sum_xi_sq = np.sum(expectations_sq**2)
        if sum_xi_sq < 1e-30:
            return float(n)
        return float(max(0.0, -np.log2(sum_xi_sq / num_paulis) - n))
    else:
        sum_xi_alpha = np.sum(expectations_sq**alpha)
        if sum_xi_alpha < 1e-30:
            return float(n)
        return float(
            max(0.0, (1 / (1 - alpha)) * np.log2(sum_xi_alpha / num_paulis) - n)
        )


def _subsystem_sre(
    state: Statevector, subsystem: list[int], alpha: int = 2
) -> float:
    """SRE for a subsystem, valid only for approximately-pure reduced states.

    The standard SRE formula conflates mixedness with magic for mixed states.
    For a pure global state, subsystem mixedness arises from entanglement,
    not local magic. We only count local magic when the subsystem is
    approximately pure.
    """
    k = len(subsystem)
    rho = state.reduced_density_matrix(subsystem)

    purity = float(np.real(np.trace(rho @ rho)))
    if purity < 1.0 - 1e-6:
        return 0.0

    num_paulis_k = 4**k
    pauli_strings_k = _all_pauli_strings(k)
    expectations_sq = np.zeros(num_paulis_k, dtype=np.float64)

    for idx, ps in enumerate(pauli_strings_k):
        P = pauli_tensor(ps)
        exp_val = np.trace(P @ rho)
        expectations_sq[idx] = abs(exp_val) ** 2

    return _sre_from_distribution(expectations_sq, num_paulis_k, k, alpha)
