"""Magic (non-stabilizerness) measures.

Implements:
- Stabilizer Rényi entropy (SRE): measures deviation from stabilizer states
  via Pauli expectation values.
- Mana: logarithmic negativity of the discrete Wigner function.
- Non-local magic: magic not attributable to individual subsystems.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector
from ..core.gates import pauli_tensor


def _all_pauli_strings(n: int) -> list[str]:
    """Generate all 4^n Pauli strings on n qubits (including identity)."""
    return ["".join(s) for s in product("IXYZ", repeat=n)]


def stabilizer_renyi_entropy(state: Statevector, alpha: int = 2) -> float:
    """Stabilizer Rényi entropy M_α(|ψ>).

    M_α = (1/(1-α)) log₂[ (1/2^n) Σ_P Tr(P|ψ><ψ|)^(2α) ] - n

    For α=2 (default):
    M_2 = -log₂[ (1/2^n) Σ_P <ψ|P|ψ>^4 ] - n

    Zero for stabilizer states, positive for magic states.
    Exponential cost O(4^n) — feasible for n ≤ ~12.
    """
    n = state.n_qubits
    num_paulis = 4**n  # total Pauli strings on n qubits

    pauli_strings = _all_pauli_strings(n)
    expectations_sq = np.zeros(num_paulis, dtype=np.float64)

    for idx, ps in enumerate(pauli_strings):
        P = pauli_tensor(ps)
        exp_val = state.expectation(P, list(range(n)))
        expectations_sq[idx] = np.real(exp_val) ** 2 + np.imag(exp_val) ** 2

    # Xi_P = |<ψ|P|ψ>|² forms a probability distribution p(P) = Xi_P / 2^n
    # SRE_α = H_α(p) - n, where H_α is the Rényi entropy of p.
    # For α=2: M_2 = -log₂[ Σ_P Xi_P² / 4^n ] - n
    if alpha == 2:
        sum_xi_sq = np.sum(expectations_sq**2)
        if sum_xi_sq < 1e-30:
            return float(n)
        m2 = -np.log2(sum_xi_sq / num_paulis) - n
        return float(max(0.0, m2))
    else:
        sum_xi_alpha = np.sum(expectations_sq**alpha)
        if sum_xi_alpha < 1e-30:
            return float(n)
        m_alpha = (1 / (1 - alpha)) * np.log2(sum_xi_alpha / num_paulis) - n
        return float(max(0.0, m_alpha))


def pauli_expectation_distribution(state: Statevector) -> NDArray[np.float64]:
    """Compute |<ψ|P|ψ>|² for all Pauli strings P.

    Returns array of 4^n values. The flatness of this distribution
    characterizes how far the state is from stabilizer states.
    """
    n = state.n_qubits
    pauli_strings = _all_pauli_strings(n)
    dist = np.zeros(len(pauli_strings), dtype=np.float64)

    for idx, ps in enumerate(pauli_strings):
        P = pauli_tensor(ps)
        exp_val = state.expectation(P, list(range(n)))
        dist[idx] = abs(exp_val) ** 2

    return dist


def mana(state: Statevector) -> float:
    """Mana (sum-negativity of discrete Wigner function).

    For odd-dimensional systems this is well-defined. For qubits, we use
    the Pauli-based approximation: mana ≈ log₂(Σ_P |<P>|) - n/2.

    This is a rough proxy; exact mana requires the discrete Wigner function.
    """
    n = state.n_qubits
    pauli_strings = _all_pauli_strings(n)

    sum_abs = 0.0
    for ps in pauli_strings:
        P = pauli_tensor(ps)
        exp_val = state.expectation(P, list(range(n)))
        sum_abs += abs(exp_val)

    # Normalize: for stabilizer states, sum of |<P>| = 2^n
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

    total_magic = stabilizer_renyi_entropy(state, alpha=2)

    # Local magic: SRE of each subsystem's reduced state
    # For mixed states, we use a proxy: SRE-like measure on the reduced state
    local_magic = 0.0
    for subsys in partition:
        local_magic += _subsystem_sre(state, subsys)

    return max(0.0, total_magic - local_magic)


def _subsystem_sre(state: Statevector, subsystem: list[int], alpha: int = 2) -> float:
    """SRE for a subsystem, valid only for approximately-pure reduced states.

    The standard SRE formula conflates mixedness with magic for mixed states
    (e.g., I/2 is a stabilizer state but the formula gives M=1). For a pure
    global state, subsystem mixedness arises from entanglement, not local magic.
    We only count local magic when the subsystem is approximately pure.
    """
    k = len(subsystem)
    rho = state.reduced_density_matrix(subsystem)

    purity = float(np.real(np.trace(rho @ rho)))
    pure_threshold = 1.0 - 1e-6
    if purity < pure_threshold:
        return 0.0

    num_paulis_k = 4**k
    pauli_strings_k = _all_pauli_strings(k)
    expectations_sq = np.zeros(num_paulis_k, dtype=np.float64)

    for idx, ps in enumerate(pauli_strings_k):
        P = pauli_tensor(ps)
        exp_val = np.trace(P @ rho)
        expectations_sq[idx] = abs(exp_val) ** 2

    if alpha == 2:
        sum_xi_sq = np.sum(expectations_sq**2)
        if sum_xi_sq < 1e-30:
            return float(k)
        m2 = -np.log2(sum_xi_sq / num_paulis_k) - k
        return float(max(0.0, m2))
    else:
        sum_xi_alpha = np.sum(expectations_sq**alpha)
        if sum_xi_alpha < 1e-30:
            return float(k)
        return float(max(0.0, (1 / (1 - alpha)) * np.log2(sum_xi_alpha / num_paulis_k) - k))
