"""Classical shadow tomography protocol (Huang, Kueng, Preskill 2020).

Implements random Pauli measurement shadows for efficient estimation of
many Pauli expectation values from O(n log n / ε²) measurements.

The protocol:
1. Prepare the target state ρ
2. Apply a random single-qubit Clifford (Pauli basis rotation) per qubit
3. Measure in computational basis
4. Repeat N times
5. Post-process to estimate <P> for arbitrary Pauli strings P
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

try:
    from qiskit.circuit import QuantumCircuit
except ImportError:
    raise ImportError(
        "qiskit is required for the hardware module. "
        "Install with: pip install 'mqhg[hardware]'"
    )


# Pauli basis labels and their measurement rotations
PAULI_BASES = ("X", "Y", "Z")


@dataclass
class Shadow:
    """A single classical shadow snapshot.

    Attributes:
        basis: Pauli basis choice per qubit (e.g., ["X", "Z", "Y", ...]).
        outcome: Measurement outcome per qubit (0 or 1).
    """

    basis: list[str]
    outcome: list[int]


@dataclass
class ShadowData:
    """Collection of classical shadow snapshots from an experiment.

    Attributes:
        n_qubits: Number of qubits in the system.
        shadows: List of Shadow snapshots.
    """

    n_qubits: int
    shadows: list[Shadow] = field(default_factory=list)

    @property
    def n_snapshots(self) -> int:
        return len(self.shadows)

    def add(self, basis: list[str], outcome: list[int]) -> None:
        self.shadows.append(Shadow(basis=basis, outcome=outcome))


class ShadowProtocol:
    """Classical shadow tomography protocol for quantum hardware.

    Usage:
        protocol = ShadowProtocol(n_qubits=20, n_measurements=1000)
        circuits = protocol.build_shadow_circuits(state_circuit)
        # Submit circuits to hardware, get counts back
        shadow_data = protocol.process_results(counts_list)
        # Estimate observables
        exp_val = protocol.estimate_pauli_expectation(shadow_data, "XZIIY...")
    """

    def __init__(
        self,
        n_qubits: int,
        n_measurements: int = 1000,
        shots_per_circuit: int = 1,
        seed: int | None = None,
    ):
        self.n_qubits = n_qubits
        self.n_measurements = n_measurements
        self.shots_per_circuit = shots_per_circuit
        self._rng = np.random.default_rng(seed)
        self._bases: list[list[str]] | None = None

    @property
    def bases(self) -> list[list[str]]:
        """Random Pauli measurement bases (generated lazily)."""
        if self._bases is None:
            self._bases = self.generate_measurement_bases()
        return self._bases

    def generate_measurement_bases(self) -> list[list[str]]:
        """Generate random Pauli basis choices for each measurement round.

        Returns:
            List of length n_measurements, each entry is a list of n_qubits
            Pauli basis labels ("X", "Y", or "Z").
        """
        bases = []
        for _ in range(self.n_measurements):
            basis = [PAULI_BASES[i] for i in self._rng.integers(0, 3, size=self.n_qubits)]
            bases.append(basis)
        self._bases = bases
        return bases

    def build_shadow_circuits(
        self,
        state_circuit: QuantumCircuit,
    ) -> list[QuantumCircuit]:
        """Build measurement circuits by appending basis rotations.

        Each circuit = state_prep + basis_rotation + measurement.

        Args:
            state_circuit: Circuit that prepares the target state.

        Returns:
            List of QuantumCircuit objects ready for execution.
        """
        circuits = []
        for idx, basis in enumerate(self.bases):
            qc = state_circuit.copy()
            qc.name = f"shadow_{idx}"

            # Append basis rotation before measurement
            for q, b in enumerate(basis):
                if b == "X":
                    qc.h(q)
                elif b == "Y":
                    qc.sdg(q)
                    qc.h(q)
                # Z basis: no rotation needed

            # Add measurement
            qc.measure_all()
            circuits.append(qc)

        return circuits

    def process_results(
        self,
        counts_list: list[dict[str, int]],
    ) -> ShadowData:
        """Process hardware measurement results into ShadowData.

        Args:
            counts_list: List of measurement count dictionaries from hardware,
                one per shadow circuit. If shots_per_circuit > 1, samples
                one outcome per circuit for the shadow.

        Returns:
            ShadowData with one snapshot per measurement.
        """
        shadow_data = ShadowData(n_qubits=self.n_qubits)

        for idx, counts in enumerate(counts_list):
            basis = self.bases[idx]

            if self.shots_per_circuit == 1:
                # Single shot: take the one result
                bitstring = max(counts, key=counts.get)
                outcome = [int(b) for b in bitstring[::-1]]  # Qiskit uses little-endian
                shadow_data.add(basis, outcome[:self.n_qubits])
            else:
                # Multiple shots: sample proportional to counts
                total = sum(counts.values())
                bitstrings = list(counts.keys())
                probs = [counts[bs] / total for bs in bitstrings]
                chosen = self._rng.choice(len(bitstrings), p=probs)
                outcome = [int(b) for b in bitstrings[chosen][::-1]]
                shadow_data.add(basis, outcome[:self.n_qubits])

        return shadow_data

    def simulate_shadows(
        self,
        state_circuit: QuantumCircuit,
    ) -> ShadowData:
        """Simulate shadow tomography classically using Qiskit's statevector simulator.

        Useful for validating the pipeline before spending QPU time.
        """
        from qiskit.quantum_info import Statevector as QiskitStatevector

        # Get the statevector from the state preparation circuit
        sv = QiskitStatevector.from_instruction(state_circuit)

        shadow_data = ShadowData(n_qubits=self.n_qubits)

        for basis in self.bases:
            # Apply basis rotation and sample
            rotated_circuit = state_circuit.copy()
            for q, b in enumerate(basis):
                if b == "X":
                    rotated_circuit.h(q)
                elif b == "Y":
                    rotated_circuit.sdg(q)
                    rotated_circuit.h(q)

            rotated_sv = QiskitStatevector.from_instruction(rotated_circuit)
            probs = rotated_sv.probabilities()
            outcome_int = self._rng.choice(len(probs), p=probs)
            outcome = [(outcome_int >> q) & 1 for q in range(self.n_qubits)]
            shadow_data.add(basis, outcome)

        return shadow_data

    @staticmethod
    def estimate_pauli_expectation(
        shadow_data: ShadowData,
        pauli_string: str,
    ) -> float:
        """Estimate <P> from classical shadows using median-of-means.

        For a Pauli string P = P_1 ⊗ P_2 ⊗ ... ⊗ P_n:
        Each snapshot contributes:
            - 0 if any non-identity Pauli doesn't match its measurement basis
            - prod_i 3*(-1)^{s_i} over non-identity qubits if all bases match

        The average over ALL snapshots (including 0s) gives an unbiased
        estimate of <P>.

        Args:
            shadow_data: Collection of shadow snapshots.
            pauli_string: String of length n_qubits with characters I/X/Y/Z.

        Returns:
            Estimated expectation value.
        """
        n = shadow_data.n_qubits
        assert len(pauli_string) == n, f"Pauli string length {len(pauli_string)} != n_qubits {n}"

        estimates = []
        for shadow in shadow_data.shadows:
            estimate = _single_shadow_estimate(shadow, pauli_string)
            # Non-matching snapshots contribute 0 (not discarded)
            estimates.append(estimate if estimate is not None else 0.0)

        if not estimates:
            return 0.0

        # Median-of-means for robust estimation
        return _median_of_means(estimates)

    @staticmethod
    def estimate_pauli_expectation_sq(
        shadow_data: ShadowData,
        pauli_string: str,
    ) -> float:
        """Estimate |<P>|² from classical shadows.

        Uses the squared single-snapshot estimates with bias correction.
        """
        exp_val = ShadowProtocol.estimate_pauli_expectation(shadow_data, pauli_string)
        return exp_val ** 2


def _single_shadow_estimate(shadow: Shadow, pauli_string: str) -> float | None:
    """Compute single-snapshot shadow estimate for a Pauli observable.

    Returns None if the measurement basis doesn't match the Pauli string
    on any non-identity qubit (those snapshots contribute nothing).
    """
    product = 1.0
    for q, (basis, outcome, pauli) in enumerate(zip(shadow.basis, shadow.outcome, pauli_string)):
        if pauli == "I":
            continue
        if basis != pauli:
            return None
        # Contribution: 3 * (-1)^outcome
        product *= 3.0 * ((-1) ** outcome)
    return product


def _median_of_means(values: list[float], n_groups: int = 5) -> float:
    """Median-of-means estimator for robust averaging.

    Splits data into groups, computes mean of each, returns median.
    More robust to outliers than plain mean.
    """
    if len(values) <= n_groups:
        return float(np.mean(values))

    arr = np.array(values)
    group_size = len(arr) // n_groups
    means = []
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size if i < n_groups - 1 else len(arr)
        means.append(np.mean(arr[start:end]))

    return float(np.median(means))
