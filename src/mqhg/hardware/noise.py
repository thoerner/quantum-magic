"""Noise-aware analysis and error mitigation for quantum hardware results.

Handles:
- Depolarizing noise correction for Pauli expectations
- Readout (SPAM) error mitigation
- Shadow-specific noise channel inversion
- Pre-flight fidelity feasibility checks
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from .shadows import ShadowData


def depolarizing_correction(
    raw_expectation: float,
    n_qubits: int,
    noise_rate: float,
) -> float:
    """Correct a Pauli expectation value for depolarizing noise.

    Under single-qubit depolarizing noise with rate p, the measured
    expectation is attenuated:
        <P>_noisy = (1 - 2p)^weight * <P>_ideal

    where weight is the number of non-identity Paulis.

    This inverts that attenuation. For the shadow protocol where we
    measure in random Pauli bases, the effective attenuation per qubit is:
        f = 1 - 2p

    Args:
        raw_expectation: Measured (noisy) expectation value.
        n_qubits: Number of non-identity qubits in the Pauli string (weight).
        noise_rate: Single-qubit depolarizing rate p ∈ [0, 0.5].

    Returns:
        Corrected expectation value.
    """
    if noise_rate <= 0:
        return raw_expectation
    if noise_rate >= 0.5:
        warnings.warn("Noise rate >= 0.5: correction is undefined (maximally mixed)")
        return 0.0

    attenuation = (1 - 2 * noise_rate) ** n_qubits
    if abs(attenuation) < 1e-15:
        return 0.0

    return raw_expectation / attenuation


def gate_noise_correction(
    raw_expectation: float,
    circuit_depth: int,
    n_2q_gates: int,
    gate_error_1q: float = 0.0005,
    gate_error_2q: float = 0.004,
) -> float:
    """Correct for accumulated gate noise throughout the circuit.

    Approximate model: each gate contributes depolarizing noise.
    Total fidelity ≈ (1-e1)^n1q * (1-e2)^n2q.
    Effective noise rate on Pauli expectations: <P>_noisy ≈ F * <P>_ideal.

    Args:
        raw_expectation: Measured expectation.
        circuit_depth: Total circuit depth.
        n_2q_gates: Number of two-qubit gates.
        gate_error_1q: Average single-qubit gate error.
        gate_error_2q: Average two-qubit gate error.
    """
    # Rough estimate: n_1q ≈ 2 * circuit_depth (one per qubit per layer)
    n_1q_est = 2 * circuit_depth
    fidelity = (1 - gate_error_1q) ** n_1q_est * (1 - gate_error_2q) ** n_2q_gates

    if fidelity < 1e-10:
        return 0.0

    return raw_expectation / fidelity


def readout_error_mitigation(
    counts: dict[str, int],
    calibration_matrix: NDArray[np.float64],
) -> dict[str, float]:
    """Mitigate readout (SPAM) errors using calibration matrix inversion.

    The calibration matrix M is defined such that:
        p_measured = M @ p_ideal

    We invert to get: p_ideal = M^{-1} @ p_measured

    Args:
        counts: Raw measurement counts {bitstring: count}.
        calibration_matrix: Square matrix of size 2^n × 2^n, where
            M[i,j] = P(measure i | prepared j).

    Returns:
        Mitigated probability distribution {bitstring: probability}.
    """
    n_bits = len(next(iter(counts)))
    dim = 2**n_bits
    total_shots = sum(counts.values())

    # Build probability vector
    p_measured = np.zeros(dim, dtype=np.float64)
    for bitstring, count in counts.items():
        idx = int(bitstring, 2)
        p_measured[idx] = count / total_shots

    # Invert calibration matrix
    try:
        M_inv = np.linalg.inv(calibration_matrix)
    except np.linalg.LinAlgError:
        M_inv = np.linalg.pinv(calibration_matrix)

    p_corrected = M_inv @ p_measured

    # Clip negative probabilities and renormalize
    p_corrected = np.clip(p_corrected, 0, None)
    total = p_corrected.sum()
    if total > 0:
        p_corrected /= total

    # Convert back to dict
    result = {}
    for idx in range(dim):
        if p_corrected[idx] > 1e-10:
            bitstring = format(idx, f"0{n_bits}b")
            result[bitstring] = float(p_corrected[idx])

    return result


def shadow_noise_correction(
    shadow_data: ShadowData,
    noise_rate: float,
) -> ShadowData:
    """Apply noise-aware shadow channel inversion.

    In the noisy shadow protocol (Koh-Grewal 2022), the classical shadow
    reconstruction formula changes from:
        ρ_snapshot = ⊗_i (3|b_i><b_i| - I)
    to:
        ρ_snapshot = ⊗_i ((3f^{-1})|b_i><b_i| - I) / (3f^{-1} - 1)

    where f = 1 - 2p is the noise channel parameter.

    For our purposes, this manifests as a rescaling of the Pauli
    expectation estimates. We don't modify the shadow data directly
    but provide the correction factor.

    This function returns the shadow data unchanged (the correction is
    applied at the estimation stage via depolarizing_correction).
    """
    # The correction is applied multiplicatively at estimation time.
    # This function exists for API completeness and documentation.
    return shadow_data


def noise_corrected_pauli_estimate(
    raw_estimate: float,
    pauli_weight: int,
    noise_rate: float = 0.0,
    gate_error_2q: float = 0.004,
    n_2q_gates: int = 0,
) -> float:
    """Combined noise correction for a Pauli expectation from shadows.

    Applies both measurement noise correction and gate noise correction.

    Args:
        raw_estimate: Raw shadow estimate of <P>.
        pauli_weight: Number of non-identity sites in P.
        noise_rate: Readout/measurement noise rate.
        gate_error_2q: Two-qubit gate error rate.
        n_2q_gates: Number of 2Q gates in the state preparation circuit.
    """
    corrected = raw_estimate

    # Measurement noise correction
    if noise_rate > 0:
        corrected = depolarizing_correction(corrected, pauli_weight, noise_rate)

    # Gate noise correction (simplified)
    if n_2q_gates > 0 and gate_error_2q > 0:
        gate_fidelity = (1 - gate_error_2q) ** n_2q_gates
        if gate_fidelity > 1e-10:
            corrected = corrected / gate_fidelity

    return corrected


def fidelity_threshold_check(
    circuit_depth: int,
    n_2q_gates: int,
    gate_fidelity_2q: float = 0.996,
    gate_fidelity_1q: float = 0.9995,
    n_qubits: int = 20,
    threshold: float = 0.1,
) -> dict:
    """Check if a circuit is feasible on given hardware.

    Estimates the expected circuit output fidelity and warns if it
    falls below a usability threshold.

    Args:
        circuit_depth: Total circuit depth.
        n_2q_gates: Number of two-qubit gates.
        gate_fidelity_2q: Two-qubit gate fidelity (e.g., 0.996 for IonQ Forte).
        gate_fidelity_1q: Single-qubit gate fidelity.
        n_qubits: Number of qubits.
        threshold: Minimum acceptable fidelity.

    Returns:
        Dict with fidelity estimate and feasibility assessment.
    """
    # Estimate total single-qubit gates (rough: ~2 per qubit per layer)
    n_1q_gates_est = n_qubits * circuit_depth

    fidelity_1q = gate_fidelity_1q ** n_1q_gates_est
    fidelity_2q = gate_fidelity_2q ** n_2q_gates
    total_fidelity = fidelity_1q * fidelity_2q

    feasible = total_fidelity >= threshold
    if not feasible:
        warnings.warn(
            f"Expected circuit fidelity {total_fidelity:.4f} is below threshold {threshold}. "
            f"Results may be dominated by noise. Consider reducing circuit depth or "
            f"using higher-fidelity hardware."
        )

    return {
        "fidelity_1q_contribution": fidelity_1q,
        "fidelity_2q_contribution": fidelity_2q,
        "total_fidelity": total_fidelity,
        "feasible": feasible,
        "n_1q_gates_est": n_1q_gates_est,
        "n_2q_gates": n_2q_gates,
        "recommendation": _hardware_recommendation(total_fidelity, n_2q_gates),
    }


def _hardware_recommendation(fidelity: float, n_2q_gates: int) -> str:
    """Suggest appropriate hardware based on expected fidelity."""
    if fidelity > 0.5:
        return "IBM Heron or IQM Garnet (adequate fidelity for shadow tomography)"
    elif fidelity > 0.2:
        return "IonQ Forte (higher fidelity needed; per-shot cost high for shadows)"
    elif fidelity > 0.05:
        return "Quantinuum H2 (deep circuit requires best available fidelity)"
    else:
        return "Circuit too deep for current hardware. Reduce depth or use error mitigation."
