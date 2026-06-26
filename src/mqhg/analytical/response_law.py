"""Analytical response law for magic-modulated hypergraph states.

This module packages closed-form and spectral results derived for the
ring + CCZ hypergraph family. All results are verified against the numerical
FWHT pipeline in ``mqhg.measures.magic`` (see tests/test_analytical.py).

Key results
-----------
1. Spectral (IPR) decomposition of SRE. For any phase-only state
   ``psi(x) = 2^{-n/2} exp(i Phi(x))`` the second Renyi SRE is

       M_2 = n - log2( sum_{f} IPR_f ),

   where ``f`` ranges over the 2^n flip masks (the X-support of a Pauli)
   and ``IPR_f = sum_{p} |W_f(p)|^4`` is the inverse participation ratio of
   the Walsh-Hadamard spectrum of the phase-derivative function
   ``g_f(x) = exp(i[Phi(x) - Phi(x XOR f)])``.

   Stabilizer states have ``g_f`` linear in ``x`` for every ``f``, so each
   Walsh spectrum is a single delta (``IPR_f = 1``), giving ``M_2 = 0``.
   Cubic (CCZ) phase terms make ``g_f`` non-linear, spreading the spectrum
   and producing ``M_2 > 0``.

2. Single-triplet closed form. For ``CCZ(theta)`` applied to ``|+++>`` the SRE
   is exactly

       M_2(theta) = 8 - log2( 7 cos^4 t + 56 cos^2 t + 84 cos t + 109 ),   t = theta.

3. Clifford invariance. SRE is invariant under Clifford gates, so the ring CZ
   edges do not affect SRE at all; SRE of the ring+CCZ family equals the SRE of
   the CCZ structure alone applied to ``|+>^n``.

4. Disjoint additivity. ``m`` CCZ triplets on disjoint qubits give
   ``M_2 = m * M_2(single triplet)``.

5. Response exponent. The numerically observed power-law exponent alpha ~ 1.2
   for the backreaction ``B(M) ~ K M^alpha`` is not fundamental: at small theta
   both the magic ``M`` and the MI deformation ``B`` scale as ``theta^2``, so the
   true local exponent is ``alpha -> 1`` (a linear law ``B = K M``). The apparent
   alpha ~ 1.2 from a global log-log fit is a curvature artifact driven by the
   non-monotonicity of SRE at large theta (SRE peaks near theta ~ 2.47 while
   B keeps growing to theta = pi).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector
from ..measures.magic import _fwht_inplace


# Coefficients of the single-triplet SRE polynomial in c = cos(theta):
#   P(c) = 7 c^4 + 56 c^2 + 84 c + 109,   M_2 = 8 - log2 P(c).
# Verified exactly against sympy and the numerical FWHT SRE.
_SINGLE_TRIPLET_COEFFS = (7.0, 0.0, 56.0, 84.0, 109.0)  # c^4, c^3, c^2, c^1, c^0


def single_triplet_polynomial(theta: float | NDArray) -> float | NDArray:
    """Polynomial ``P(cos theta) = 7 c^4 + 56 c^2 + 84 c + 109``.

    Equals ``sum_P |<P>|^4`` for the 3-qubit state ``CCZ(theta)|+++>``.
    At theta=0 it equals 256 = 4^3 * 4^0 ... specifically P(1)=256 giving M_2=0.
    """
    c = np.cos(theta)
    a4, a3, a2, a1, a0 = _SINGLE_TRIPLET_COEFFS
    return a4 * c**4 + a3 * c**3 + a2 * c**2 + a1 * c + a0


def sre_single_triplet(theta: float | NDArray) -> float | NDArray:
    """Closed-form second Renyi SRE for ``CCZ(theta)|+++>``.

        M_2(theta) = 8 - log2( 7 cos^4 t + 56 cos^2 t + 84 cos t + 109 ).

    Returns 0 at theta=0 (stabilizer) and rises to a maximum near theta ~ 2.5.
    """
    poly = single_triplet_polynomial(theta)
    return 8.0 - np.log2(poly)


def sre_disjoint_triplets(theta: float | NDArray, m: int) -> float | NDArray:
    """SRE for ``m`` CCZ triplets acting on disjoint qubit triples.

    By tensor-product additivity of SRE, this is ``m * M_2(single triplet)``.
    Requires the triplets to be vertex-disjoint (no shared qubits).
    """
    if m < 0:
        raise ValueError("m must be non-negative")
    return m * sre_single_triplet(theta)


def sre_ipr_decomposition(state: Statevector) -> tuple[float, NDArray[np.float64]]:
    """Spectral (IPR) decomposition of SRE for a phase-only state.

    Returns ``(M_2, ipr_per_flip_mask)`` where

        M_2 = n - log2( sum_f IPR_f )

    and ``ipr_per_flip_mask[f] = sum_p |<P_{f,p}>|^4`` is the inverse
    participation ratio of the Walsh spectrum of the phase-derivative for
    flip mask ``f``. Stabilizer contributions have ``IPR_f = 1``; values below
    1 indicate spectral spreading caused by non-Clifford (e.g. CCZ) phases.

    This is exact for any state (not only phase-only ones); the interpretation
    in terms of phase derivatives is exact when amplitudes have uniform modulus.

    Cost: O(n * 4^n) time, O(2^n) memory -- same order as the direct SRE.
    """
    psi = state.amplitudes
    n = state.n_qubits
    dim = 1 << n
    indices = np.arange(dim, dtype=np.int64)

    ipr = np.zeros(dim, dtype=np.float64)
    for fm in range(dim):
        base = np.conj(psi[indices ^ fm]) * psi
        _fwht_inplace(base)
        w = np.real(base * np.conj(base))  # |<P_{fm,pm}>|^2 over all pm
        ipr[fm] = np.sum(w * w)

    total = float(np.sum(ipr))
    if total < 1e-30:
        return float(n), ipr
    m2 = max(0.0, n - np.log2(total))
    return float(m2), ipr


def local_response_exponent(
    magic: NDArray[np.float64],
    backreaction: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Local power-law exponent ``alpha(theta) = d log B / d log M``.

    Computed by finite differences of the log-transformed curves. A constant
    value of 1 indicates a genuinely linear response ``B = K M``; deviations
    reveal where the global log-log fit is distorted by curvature.

    Args:
        magic: Magic values M(theta) along a sweep (must be positive).
        backreaction: Backreaction values B(theta) along the same sweep.

    Returns:
        Array of local exponents (same length as inputs, minus masked points).
    """
    magic = np.asarray(magic, dtype=np.float64)
    backreaction = np.asarray(backreaction, dtype=np.float64)
    mask = (magic > 1e-9) & (backreaction > 1e-9)
    log_m = np.log(magic[mask])
    log_b = np.log(backreaction[mask])
    if len(log_m) < 2:
        return np.array([])
    d_log_m = np.gradient(log_m)
    d_log_b = np.gradient(log_b)
    with np.errstate(divide="ignore", invalid="ignore"):
        alpha = np.where(np.abs(d_log_m) > 1e-12, d_log_b / d_log_m, np.nan)
    return alpha
