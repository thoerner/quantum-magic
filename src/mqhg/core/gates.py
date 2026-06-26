"""Quantum gate definitions for qubit systems.

Includes Clifford gates (Pauli, H, CZ, CNOT, S) and non-Clifford gates
(T, CCZ, controlled-phase) needed for magic injection.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# --- Single-qubit gates ---

pauli_x: NDArray[np.complex128] = np.array([[0, 1], [1, 0]], dtype=np.complex128)
pauli_y: NDArray[np.complex128] = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
pauli_z: NDArray[np.complex128] = np.array([[1, 0], [0, -1]], dtype=np.complex128)
hadamard: NDArray[np.complex128] = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
identity_2: NDArray[np.complex128] = np.eye(2, dtype=np.complex128)

# S gate (phase gate, Clifford)
s_gate: NDArray[np.complex128] = np.array([[1, 0], [0, 1j]], dtype=np.complex128)

# T gate (pi/8 gate, non-Clifford, injects magic)
t_gate: NDArray[np.complex128] = np.array(
    [[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128
)


def rz(theta: float) -> NDArray[np.complex128]:
    """Rotation about Z axis by angle theta."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]],
        dtype=np.complex128,
    )


def phase_gate(phi: float) -> NDArray[np.complex128]:
    """General phase gate diag(1, e^{i*phi})."""
    return np.array([[1, 0], [0, np.exp(1j * phi)]], dtype=np.complex128)


# --- Two-qubit gates ---

# CZ gate (Clifford): diagonal, applies -1 to |11>
cz: NDArray[np.complex128] = np.diag([1, 1, 1, -1]).astype(np.complex128)

# CNOT gate (Clifford)
cnot: NDArray[np.complex128] = np.array(
    [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.complex128
)


def controlled_phase(phi: float) -> NDArray[np.complex128]:
    """Two-qubit controlled phase gate: diag(1, 1, 1, e^{i*phi}).

    phi=pi gives CZ. Non-Clifford for generic phi.
    """
    return np.diag([1, 1, 1, np.exp(1j * phi)]).astype(np.complex128)


# --- Three-qubit gates ---

# CCZ gate (non-Clifford): applies -1 phase to |111>
ccz: NDArray[np.complex128] = np.diag([1, 1, 1, 1, 1, 1, 1, -1]).astype(np.complex128)

# Toffoli (CCNOT)
_toffoli = np.eye(8, dtype=np.complex128)
_toffoli[6, 6] = 0
_toffoli[7, 7] = 0
_toffoli[6, 7] = 1
_toffoli[7, 6] = 1
toffoli: NDArray[np.complex128] = _toffoli


def controlled_controlled_phase(phi: float) -> NDArray[np.complex128]:
    """Three-qubit controlled-controlled-phase: diag(1,...,1, e^{i*phi}).

    phi=pi gives CCZ. Non-Clifford for generic phi.
    """
    d = np.ones(8, dtype=np.complex128)
    d[7] = np.exp(1j * phi)
    return np.diag(d)


def multi_controlled_phase(n_controls: int, phi: float) -> NDArray[np.complex128]:
    """N-qubit controlled phase gate: applies e^{i*phi} to |11...1>.

    Acts on (n_controls + 0) qubits total (phase on all-ones state).
    For hyperedge phase gates on k qubits.
    """
    dim = 2**n_controls
    d = np.ones(dim, dtype=np.complex128)
    d[-1] = np.exp(1j * phi)
    return np.diag(d)


# --- Pauli string operations ---

PAULI_MATRICES = {
    "I": identity_2,
    "X": pauli_x,
    "Y": pauli_y,
    "Z": pauli_z,
}


def pauli_tensor(pauli_string: str) -> NDArray[np.complex128]:
    """Construct tensor product of Pauli matrices from string like 'XZIY'."""
    result = np.array([[1]], dtype=np.complex128)
    for char in pauli_string:
        result = np.kron(result, PAULI_MATRICES[char.upper()])
    return result
