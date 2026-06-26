"""Model 2: HaPPY-code perturbation model.

Implements a small holographic tensor-network code inspired by the
Pastawski-Yoshida-Harlow-Preskill (HaPPY) pentagon code.

The HaPPY code uses perfect tensors on a hyperbolic tiling to model
bulk/boundary AdS/CFT correspondence. We build a minimal version and
study perturbations from stabilizer to non-stabilizer tensors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector
from ..core.gates import cz, ccz, t_gate, hadamard


class TensorType(Enum):
    PERFECT_STABILIZER = "perfect_stabilizer"
    NON_STABILIZER = "non_stabilizer"
    RANDOM = "random"


@dataclass
class HaPPYConfig:
    """Configuration for the HaPPY code model.

    A minimal HaPPY-like code with:
    - n_bulk: number of bulk (logical) qubits
    - n_boundary: number of boundary (physical) qubits
    - n_layers: depth of the tensor network (hyperbolic layers)
    """

    n_bulk: int = 1
    n_boundary: int = 5
    n_layers: int = 1
    tensor_type: TensorType = TensorType.PERFECT_STABILIZER


def _perfect_tensor_5qubit() -> NDArray[np.complex128]:
    """Construct the [[5,1,3]] perfect tensor (encoding isometry).

    The 5-qubit code encodes 1 logical qubit into 5 physical qubits.
    It is the smallest perfect quantum error-correcting code and forms
    the basis of the HaPPY pentagon model.

    Returns the 2x32 encoding isometry (maps 1 logical qubit to 5 physical).
    """
    # [[5,1,3]] code stabilizers: XZZXI, IXZZX, XIXZZ, ZXIXZ
    # Logical operators: X_L = XXXXX, Z_L = ZZZZZ
    # Codewords:
    # |0_L> = (|00000> + |10010> + |01001> + |10100> + |01010>
    #          + |11011> + |00110> + |11101> + |00101> + |11110>
    #          + |01111> + |10001> + |01100> + |10111> + |11000>
    #          + |00011>) / 4
    # |1_L> = X_L |0_L>

    dim = 32  # 2^5
    zero_L = np.zeros(dim, dtype=np.complex128)
    one_L = np.zeros(dim, dtype=np.complex128)

    # |0_L> codeword indices (from stabilizer code construction)
    zero_indices = [0, 18, 9, 20, 10, 27, 6, 29, 5, 30, 15, 17, 12, 23, 24, 3]
    for idx in zero_indices:
        zero_L[idx] = 1.0 / 4.0

    # |1_L> = X^⊗5 |0_L> (bit-flip all)
    for idx in zero_indices:
        one_L[31 - idx] = 1.0 / 4.0

    # Encoding isometry: V = |0_L><0| + |1_L><1|
    V = np.zeros((dim, 2), dtype=np.complex128)
    V[:, 0] = zero_L
    V[:, 1] = one_L

    return V


class HaPPYCode:
    """Minimal HaPPY-like holographic tensor network code.

    For the research, the key operations are:
    1. Encode bulk logical state → boundary physical state
    2. Perturb: replace stabilizer tensors with non-stabilizer ones
    3. Measure: entropy, min-cut, reconstruction fidelity
    """

    def __init__(self, config: HaPPYConfig | None = None):
        self.config = config or HaPPYConfig()
        self._encoder = None

    @property
    def encoder(self) -> NDArray[np.complex128]:
        """Encoding isometry from bulk to boundary."""
        if self._encoder is None:
            self._encoder = self._build_encoder()
        return self._encoder

    def _build_encoder(self) -> NDArray[np.complex128]:
        """Build encoding map based on tensor type."""
        if self.config.tensor_type == TensorType.PERFECT_STABILIZER:
            return _perfect_tensor_5qubit()
        elif self.config.tensor_type == TensorType.NON_STABILIZER:
            return self._non_stabilizer_encoder()
        else:
            return self._random_encoder()

    def _non_stabilizer_encoder(self) -> NDArray[np.complex128]:
        """Perturbed encoder with non-Clifford (T-gate) modification."""
        V = _perfect_tensor_5qubit()
        # Apply T gate to the first physical qubit of each codeword
        T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)
        # T acts on qubit 0 of the 5-qubit space
        T_full = np.kron(T, np.eye(16, dtype=np.complex128))
        return T_full @ V

    def _random_encoder(self) -> NDArray[np.complex128]:
        """Random isometry (no geometric structure)."""
        rng = np.random.default_rng(42)
        dim = 2**self.config.n_boundary
        V = rng.standard_normal((dim, 2)) + 1j * rng.standard_normal((dim, 2))
        Q, _ = np.linalg.qr(V)
        return Q[:, :2]

    def encode(self, logical_state: NDArray[np.complex128]) -> Statevector:
        """Encode a logical qubit state into boundary physical qubits.

        Args:
            logical_state: 2-component vector [alpha, beta] for α|0>+β|1>.

        Returns:
            Statevector on n_boundary qubits.
        """
        logical = np.asarray(logical_state, dtype=np.complex128)
        assert logical.shape == (2,)
        physical = self.encoder @ logical
        return Statevector(physical)

    def decode_fidelity(self, boundary_state: Statevector, target_logical: NDArray[np.complex128]) -> float:
        """Measure how well the boundary state decodes to the target logical state.

        Returns fidelity F = |<target|decoded>|².
        """
        target = np.asarray(target_logical, dtype=np.complex128)
        # Project boundary state back through encoder
        decoded = self.encoder.conj().T @ boundary_state.amplitudes
        decoded /= np.linalg.norm(decoded)
        return float(abs(np.vdot(target, decoded)) ** 2)

    def boundary_entropy(self, logical_state: NDArray[np.complex128], subsystem: list[int]) -> float:
        """Compute entanglement entropy of boundary subsystem after encoding."""
        from ..measures.entanglement import subsystem_entropy
        state = self.encode(logical_state)
        return subsystem_entropy(state, subsystem)

    def min_cut_proxy(self, logical_state: NDArray[np.complex128]) -> float:
        """Proxy for the RT minimal surface: minimum entropy over bipartitions.

        In the HaPPY code, the RT surface corresponds to the minimum cut
        through the tensor network that separates a boundary region from
        its complement.
        """
        from ..measures.entanglement import subsystem_entropy
        state = self.encode(logical_state)
        n = self.config.n_boundary

        min_entropy = float("inf")
        for size in range(1, n // 2 + 1):
            from itertools import combinations
            for subset in combinations(range(n), size):
                s = subsystem_entropy(state, list(subset))
                min_entropy = min(min_entropy, s)

        return min_entropy

    def compare_stabilizer_vs_magic(
        self,
        logical_state: NDArray[np.complex128] | None = None,
    ) -> dict:
        """Compare stabilizer and non-stabilizer encoders on key observables.

        Returns dict with results for both tensor types.
        """
        from ..measures.magic import stabilizer_renyi_entropy, nonlocal_magic
        from ..measures.entanglement import mutual_information_matrix

        if logical_state is None:
            logical_state = np.array([1, 0], dtype=np.complex128)

        results = {}
        for tt in [TensorType.PERFECT_STABILIZER, TensorType.NON_STABILIZER]:
            self.config.tensor_type = tt
            self._encoder = None  # Force rebuild

            state = self.encode(logical_state)
            results[tt.value] = {
                "sre": stabilizer_renyi_entropy(state),
                "nonlocal_magic": nonlocal_magic(state),
                "mi_matrix": mutual_information_matrix(state),
                "min_cut": self.min_cut_proxy(logical_state),
            }

        return results
