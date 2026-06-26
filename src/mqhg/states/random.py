"""Random state generation: Clifford circuits, magic circuits, Haar random."""

from __future__ import annotations

import numpy as np

from ..core.statevector import Statevector
from ..core.gates import (
    hadamard, s_gate, cz, cnot, t_gate, ccz,
    pauli_x, pauli_y, pauli_z,
)


# Single-qubit Clifford generators (H, S, and their products generate the
# single-qubit Clifford group up to phases)
_CLIFFORD_1Q = [hadamard, s_gate, hadamard @ s_gate, s_gate @ hadamard,
                hadamard @ s_gate @ hadamard, s_gate @ hadamard @ s_gate]


def random_clifford_state(n_qubits: int, depth: int, rng: np.random.Generator | None = None) -> Statevector:
    """Generate a random state using only Clifford gates (zero magic).

    Alternates random single-qubit Cliffords with random CZ/CNOT entangling layers.
    """
    if rng is None:
        rng = np.random.default_rng()

    state = Statevector.plus_n(n_qubits)

    for _ in range(depth):
        # Single-qubit Clifford layer
        for q in range(n_qubits):
            gate = _CLIFFORD_1Q[rng.integers(len(_CLIFFORD_1Q))]
            state = state.apply_gate(gate, [q])

        # Entangling layer: random subset of CZ gates
        pairs = list(range(n_qubits))
        rng.shuffle(pairs)
        for i in range(0, n_qubits - 1, 2):
            if rng.random() < 0.5:
                state = state.apply_gate(cz, [pairs[i], pairs[i + 1]])

    return state


def random_magic_state(
    n_qubits: int,
    depth: int,
    p_magic: float = 0.1,
    rng: np.random.Generator | None = None,
) -> Statevector:
    """Generate a random state with tunable magic density.

    Uses Clifford + sparse T/CCZ gates. p_magic controls the probability
    of inserting a non-Clifford gate at each opportunity.

    Args:
        n_qubits: Number of qubits.
        depth: Circuit depth (number of layers).
        p_magic: Probability of non-Clifford gate per slot.
        rng: Random number generator.
    """
    if rng is None:
        rng = np.random.default_rng()

    state = Statevector.plus_n(n_qubits)

    for _ in range(depth):
        # Single-qubit layer: Clifford or T gate
        for q in range(n_qubits):
            if rng.random() < p_magic:
                state = state.apply_gate(t_gate, [q])
            else:
                gate = _CLIFFORD_1Q[rng.integers(len(_CLIFFORD_1Q))]
                state = state.apply_gate(gate, [q])

        # Entangling layer
        pairs = list(range(n_qubits))
        rng.shuffle(pairs)
        for i in range(0, n_qubits - 1, 2):
            state = state.apply_gate(cz, [pairs[i], pairs[i + 1]])

        # Optional CCZ layer for 3-body magic
        if n_qubits >= 3 and rng.random() < p_magic:
            targets = sorted(rng.choice(n_qubits, size=3, replace=False))
            state = state.apply_gate(ccz, targets)

    return state


def haar_random_state(n_qubits: int, rng: np.random.Generator | None = None) -> Statevector:
    """Haar-random state (maximal magic/complexity)."""
    if rng is None:
        rng = np.random.default_rng()

    dim = 2**n_qubits
    real = rng.standard_normal(dim)
    imag = rng.standard_normal(dim)
    amps = (real + 1j * imag) / np.sqrt(2)
    amps /= np.linalg.norm(amps)
    return Statevector(amps)
