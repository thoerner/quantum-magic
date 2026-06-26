"""N-qubit statevector representation and manipulation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


class Statevector:
    """Dense statevector for n-qubit systems.

    Stores amplitudes as a complex vector of length 2^n.
    Supports gate application, partial trace, and reduced density matrices.
    """

    def __init__(self, amplitudes: NDArray[np.complex128]):
        n_amplitudes = len(amplitudes)
        if n_amplitudes == 0 or (n_amplitudes & (n_amplitudes - 1)) != 0:
            raise ValueError(f"Amplitude vector length must be a power of 2, got {n_amplitudes}")
        self._amps = np.asarray(amplitudes, dtype=np.complex128)
        self._n = int(np.log2(n_amplitudes))

    @classmethod
    def plus_n(cls, n: int) -> Statevector:
        """|+>^n: uniform superposition over all computational basis states."""
        amps = np.full(2**n, 1.0 / np.sqrt(2**n), dtype=np.complex128)
        return cls(amps)

    @classmethod
    def zero_n(cls, n: int) -> Statevector:
        """|0>^n: all-zeros computational basis state."""
        amps = np.zeros(2**n, dtype=np.complex128)
        amps[0] = 1.0
        return cls(amps)

    @classmethod
    def computational_basis(cls, n: int, index: int) -> Statevector:
        """Computational basis state |index> for n qubits."""
        amps = np.zeros(2**n, dtype=np.complex128)
        amps[index] = 1.0
        return cls(amps)

    @property
    def n_qubits(self) -> int:
        return self._n

    @property
    def amplitudes(self) -> NDArray[np.complex128]:
        return self._amps

    @property
    def dim(self) -> int:
        return 2**self._n

    def copy(self) -> Statevector:
        return Statevector(self._amps.copy())

    def norm(self) -> float:
        return float(np.linalg.norm(self._amps))

    def normalize(self) -> Statevector:
        """Return normalized copy."""
        n = self.norm()
        if n < 1e-15:
            raise ValueError("Cannot normalize zero vector")
        return Statevector(self._amps / n)

    def inner(self, other: Statevector) -> complex:
        return complex(np.vdot(self._amps, other._amps))

    def probabilities(self) -> NDArray[np.float64]:
        return np.abs(self._amps) ** 2

    def apply_gate(self, gate: NDArray[np.complex128], targets: list[int]) -> Statevector:
        """Apply a gate matrix to specified qubit indices.

        Args:
            gate: Unitary matrix of shape (2^k, 2^k) for k target qubits.
            targets: List of qubit indices the gate acts on (0-indexed, MSB ordering).

        Returns:
            New Statevector with gate applied.
        """
        n = self._n
        k = len(targets)
        assert gate.shape == (2**k, 2**k), f"Gate shape {gate.shape} incompatible with {k} targets"

        # Reshape statevector into tensor of shape (2,2,...,2)
        psi = self._amps.reshape([2] * n)

        # Move target qubits to the front
        axes_order = targets + [i for i in range(n) if i not in targets]
        psi = np.transpose(psi, axes_order)

        # Reshape so target qubits are one index
        psi = psi.reshape(2**k, 2**(n - k))

        # Apply gate
        psi = gate @ psi

        # Reshape back
        psi = psi.reshape([2] * n)

        # Invert the permutation
        inv_order = [0] * n
        for new_pos, old_pos in enumerate(axes_order):
            inv_order[old_pos] = new_pos
        psi = np.transpose(psi, inv_order)

        return Statevector(psi.reshape(2**n))

    def reduced_density_matrix(self, subsystem: list[int]) -> NDArray[np.complex128]:
        """Compute reduced density matrix for the specified qubit subsystem.

        Args:
            subsystem: List of qubit indices to keep (trace out the rest).

        Returns:
            Density matrix of shape (2^|subsystem|, 2^|subsystem|).
        """
        n = self._n
        k = len(subsystem)
        complement = [i for i in range(n) if i not in subsystem]

        # Reshape to tensor
        psi = self._amps.reshape([2] * n)

        # Reorder: subsystem qubits first, then complement
        axes_order = subsystem + complement
        psi = np.transpose(psi, axes_order)

        # Reshape to (2^k, 2^(n-k))
        psi = psi.reshape(2**k, 2**(n - k))

        # rho_A = Tr_B(|psi><psi|) = psi @ psi^dagger (with appropriate reshape)
        rho = psi @ psi.conj().T

        return rho

    def expectation(self, operator: NDArray[np.complex128], targets: list[int]) -> complex:
        """Compute <psi|O|psi> for operator O acting on target qubits."""
        result = self.apply_gate(operator, targets)
        return self.inner(result)

    def __repr__(self) -> str:
        return f"Statevector(n_qubits={self._n}, norm={self.norm():.6f})"
