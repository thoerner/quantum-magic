"""Backreaction observables.

Measures whether localized excitations deform emergent geometry,
and whether that deformation correlates with non-local magic.

Key quantity:
    B_i = ||D_after - D_before|| around excitation site i

The central test:
    Does B_i remain near zero for stabilizer-only states?
    Does B_i increase with non-local magic?
    Does B_i become incoherent for highly random states?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector
from ..core.gates import pauli_x, pauli_z, t_gate, phase_gate, rz
from ..measures.entanglement import mutual_information_matrix
from ..measures.geometry import mutual_info_distance_matrix, emergent_metric
from ..measures.magic import stabilizer_renyi_entropy, nonlocal_magic


@dataclass
class BackreactionResult:
    """Results from a backreaction measurement."""

    site: int
    excitation_type: str
    distance_change_frobenius: float
    distance_change_local: float
    mi_change_frobenius: float
    sre_before: float
    sre_after: float
    nonlocal_magic_before: float
    nonlocal_magic_after: float


class BackreactionObservable:
    """Compute backreaction observables for a given state and excitation."""

    def __init__(self, state: Statevector):
        self.state = state
        self.n = state.n_qubits

    def measure(
        self,
        site: int,
        excitation_type: str = "X",
        epsilon: float = 0.1,
    ) -> BackreactionResult:
        """Measure backreaction from a localized excitation.

        Args:
            site: Qubit index to excite.
            excitation_type: One of "X", "Z", "T", or a float (phase angle).
            epsilon: Strength parameter for continuous excitations.
        """
        state_before = self.state

        # Apply excitation
        gate = self._get_excitation_gate(excitation_type, epsilon)
        state_after = state_before.apply_gate(gate, [site])

        # Distance matrices
        dist_before = mutual_info_distance_matrix(state_before)
        dist_after = mutual_info_distance_matrix(state_after)

        # MI matrices
        mi_before = mutual_information_matrix(state_before)
        mi_after = mutual_information_matrix(state_after)

        # Global change
        dist_change_fro = float(np.linalg.norm(dist_after - dist_before, "fro"))

        # Local change (rows/cols involving the excited qubit)
        local_diff = dist_after[site, :] - dist_before[site, :]
        dist_change_local = float(np.linalg.norm(local_diff))

        # MI change
        mi_change_fro = float(np.linalg.norm(mi_after - mi_before, "fro"))

        # Magic before/after
        sre_before = stabilizer_renyi_entropy(state_before)
        sre_after = stabilizer_renyi_entropy(state_after)
        nl_before = nonlocal_magic(state_before)
        nl_after = nonlocal_magic(state_after)

        return BackreactionResult(
            site=site,
            excitation_type=excitation_type,
            distance_change_frobenius=dist_change_fro,
            distance_change_local=dist_change_local,
            mi_change_frobenius=mi_change_fro,
            sre_before=sre_before,
            sre_after=sre_after,
            nonlocal_magic_before=nl_before,
            nonlocal_magic_after=nl_after,
        )

    def measure_all_sites(self, excitation_type: str = "X") -> list[BackreactionResult]:
        """Measure backreaction at every qubit site."""
        return [self.measure(site, excitation_type) for site in range(self.n)]

    def response_coefficient(
        self,
        site: int,
        n_strengths: int = 10,
    ) -> float:
        """Estimate linear response coefficient dB/dε at small ε.

        This tests Conjecture D: whether backreaction is proportional to
        excitation strength in a linear-response regime.
        """
        epsilons = np.linspace(0.01, 0.5, n_strengths)
        backreactions = []

        for eps in epsilons:
            gate = rz(eps)
            state_after = self.state.apply_gate(gate, [site])
            dist_before = mutual_info_distance_matrix(self.state)
            dist_after = mutual_info_distance_matrix(state_after)
            br = float(np.linalg.norm(dist_after - dist_before, "fro"))
            backreactions.append(br)

        # Linear fit: B ≈ K * ε
        coeffs = np.polyfit(epsilons, backreactions, 1)
        return float(coeffs[0])  # slope = response coefficient K

    @staticmethod
    def _get_excitation_gate(excitation_type: str, epsilon: float) -> NDArray[np.complex128]:
        if excitation_type == "X":
            return pauli_x
        elif excitation_type == "Z":
            return pauli_z
        elif excitation_type == "T":
            return t_gate
        elif excitation_type == "Rz":
            return rz(epsilon)
        else:
            try:
                angle = float(excitation_type)
                return phase_gate(angle)
            except ValueError:
                raise ValueError(f"Unknown excitation type: {excitation_type}")
