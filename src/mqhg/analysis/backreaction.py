"""Backreaction observables.

Measures geometric deformation from magic injection. In the holographic
picture, stabilizer states have flat (trivial) emergent geometry, while
magic creates non-trivial geometry. "Backreaction" quantifies this
geometric deformation relative to a stabilizer reference.

Key quantity:
    B = ||MI(ψ_magic) - MI(ψ_stabilizer)||_F

The central test:
    Does B remain near zero for stabilizer-only states?
    Does B increase with non-local magic?
    Is the deformation structured (not random)?
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector
from ..core.gates import pauli_x, pauli_z, t_gate, phase_gate, rz
from ..measures.entanglement import mutual_information_matrix
from ..measures.geometry import mutual_info_distance_matrix, correlator_distance_matrix, emergent_metric
from ..measures.magic import stabilizer_renyi_entropy, nonlocal_magic, _fast_sre_direct, _subsystem_sre


@dataclass
class BackreactionResult:
    """Results from a backreaction measurement."""

    mi_deformation: float
    mi_max: float
    mi_mean: float
    sre: float
    nonlocal_magic_val: float


class BackreactionObservable:
    """Compute backreaction as geometric deformation from stabilizer reference."""

    def __init__(self, state: Statevector, reference: Statevector | None = None):
        """
        Args:
            state: The state to measure.
            reference: Stabilizer reference state. If None, uses flat (zero MI)
                      as reference (appropriate for highly-entangled stabilizer
                      states like graph states that have no pairwise MI).
        """
        self.state = state
        self.n = state.n_qubits
        if reference is not None:
            self.mi_ref = mutual_information_matrix(reference)
        else:
            self.mi_ref = np.zeros((self.n, self.n), dtype=np.float64)

    def measure(self) -> BackreactionResult:
        """Measure geometric deformation from the stabilizer reference.

        Backreaction = how much pairwise MI structure the state has
        relative to the stabilizer baseline (which has flat/zero MI).
        """
        mi = mutual_information_matrix(self.state)
        mi_diff = mi - self.mi_ref

        mi_deformation = float(np.linalg.norm(mi_diff, "fro"))
        mi_max = float(mi.max())
        mi_mean = float(mi[np.triu_indices(self.n, k=1)].mean())

        # Compute total SRE once, then derive non-local magic from it
        sre = _fast_sre_direct(self.state.amplitudes, self.n, alpha=2)
        local_magic = sum(
            _subsystem_sre(self.state, [i]) for i in range(self.n)
        )
        nl = max(0.0, sre - local_magic)

        return BackreactionResult(
            mi_deformation=mi_deformation,
            mi_max=mi_max,
            mi_mean=mi_mean,
            sre=sre,
            nonlocal_magic_val=nl,
        )

    def response_coefficient(
        self,
        n_strengths: int = 10,
    ) -> float:
        """Estimate linear response: how MI deformation scales with magic.

        Uses Rz(ε) applied to all qubits simultaneously as a magic injection
        mechanism, measuring how MI structure emerges with increasing ε.
        """
        epsilons = np.linspace(0.01, 0.5, n_strengths)
        deformations = []

        for eps in epsilons:
            gate = rz(eps)
            state_perturbed = self.state
            for q in range(self.n):
                state_perturbed = state_perturbed.apply_gate(gate, [q])
            mi = mutual_information_matrix(state_perturbed)
            deformation = float(np.linalg.norm(mi - self.mi_ref, "fro"))
            deformations.append(deformation)

        coeffs = np.polyfit(epsilons, deformations, 1)
        return float(coeffs[0])
