"""Model 3: Approximate holographic code with magic.

Exact stabilizer/perfect codes are too rigid. This model explores
approximate quantum error-correcting codes where:
- Encoding is not exact (finite-distance effects)
- Tensors are not perfectly stabilizer
- Magic controls the quality of bulk-boundary correspondence
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector
from ..core.gates import hadamard, cz, t_gate


@dataclass
class ApproximateCodeConfig:
    """Configuration for approximate holographic codes."""

    n_physical: int = 6
    n_logical: int = 1
    noise_strength: float = 0.0
    magic_strength: float = 0.0
    seed: int = 42


class ApproximateHolographicCode:
    """Approximate holographic code interpolating between stabilizer and non-stabilizer.

    The key idea: parameterize a family of codes from exact stabilizer (rigid, no
    backreaction) through approximate non-stabilizer (deformable, with backreaction).

    magic_strength=0 → perfect stabilizer code (rigid geometry)
    magic_strength>0 → approximate code with non-Clifford structure
    """

    def __init__(self, config: ApproximateCodeConfig | None = None):
        self.config = config or ApproximateCodeConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def build_encoder(self) -> NDArray[np.complex128]:
        """Construct parameterized encoding isometry.

        Starts from a stabilizer encoder and applies controlled
        non-Clifford perturbation.
        """
        n_p = self.config.n_physical
        n_l = self.config.n_logical
        dim_p = 2**n_p
        dim_l = 2**n_l

        # Base: random stabilizer-like encoder (Clifford circuit applied to |0...0>|ψ_L>)
        # For simplicity, use a structured encoder from random Clifford unitaries
        V_base = self._clifford_encoder(dim_p, dim_l)

        # Non-stabilizer perturbation
        if self.config.magic_strength > 0:
            V_perturbed = self._apply_magic_perturbation(V_base)
            # Re-orthogonalize
            Q, _ = np.linalg.qr(V_perturbed)
            return Q[:, :dim_l]

        return V_base

    def _clifford_encoder(self, dim_p: int, dim_l: int) -> NDArray[np.complex128]:
        """Generate a stabilizer-compatible encoding isometry."""
        # Use a fixed structure: Hadamard + CZ layers
        n_p = self.config.n_physical
        state_vecs = []

        for logical_idx in range(dim_l):
            # Start with computational basis for logical
            psi = np.zeros(dim_p, dtype=np.complex128)
            psi[logical_idx] = 1.0

            # Apply encoding circuit (Hadamard on all + CZ ladder)
            psi = psi.reshape([2] * n_p)

            # Hadamard on non-logical qubits
            for q in range(dim_l.bit_length(), n_p):
                psi = np.swapaxes(psi, 0, q)
                shape = psi.shape
                psi_2d = psi.reshape(2, -1)
                psi_2d = hadamard @ psi_2d
                psi = psi_2d.reshape(shape)
                psi = np.swapaxes(psi, 0, q)

            state_vecs.append(psi.flatten())

        V = np.column_stack(state_vecs)

        # Orthogonalize
        Q, _ = np.linalg.qr(V)
        return Q[:, :dim_l]

    def _apply_magic_perturbation(self, V: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Apply non-Clifford perturbation controlled by magic_strength."""
        dim_p = V.shape[0]
        n_p = int(np.log2(dim_p))
        eps = self.config.magic_strength

        # Construct a non-Clifford unitary as exp(-i*eps*H_magic)
        # where H_magic has T-gate-like structure
        H_magic = np.zeros((dim_p, dim_p), dtype=np.complex128)
        for q in range(n_p):
            # Diagonal contribution from T-like rotation on qubit q
            for idx in range(dim_p):
                if (idx >> (n_p - 1 - q)) & 1:
                    H_magic[idx, idx] += np.pi / 4

        U_magic = np.diag(np.exp(-1j * eps * np.diag(H_magic)))
        return U_magic @ V

    def encode(self, logical_state: NDArray[np.complex128]) -> Statevector:
        """Encode logical state."""
        V = self.build_encoder()
        physical = V @ np.asarray(logical_state, dtype=np.complex128)
        return Statevector(physical)

    def reconstruction_error(
        self,
        logical_state: NDArray[np.complex128],
        erased_qubits: list[int],
    ) -> float:
        """Measure reconstruction error when some physical qubits are lost.

        In perfect codes, any subset smaller than the code distance can be
        reconstructed exactly. In approximate codes, reconstruction degrades
        smoothly — this degradation may relate to backreaction.
        """
        state = self.encode(logical_state)
        n_p = self.config.n_physical

        kept = [q for q in range(n_p) if q not in erased_qubits]
        rho_kept = state.reduced_density_matrix(kept)

        # Ideal: encode and trace same qubits
        config_ideal = ApproximateCodeConfig(
            n_physical=self.config.n_physical,
            n_logical=self.config.n_logical,
            magic_strength=0.0,
            seed=self.config.seed,
        )
        ideal_code = ApproximateHolographicCode(config_ideal)
        ideal_state = ideal_code.encode(logical_state)
        rho_ideal = ideal_state.reduced_density_matrix(kept)

        # Trace distance proxy: ||ρ - ρ_ideal||_1 / 2
        diff = rho_kept - rho_ideal
        singular_values = np.linalg.svd(diff, compute_uv=False)
        return float(np.sum(singular_values) / 2)

    def sweep_magic_strength(
        self,
        logical_state: NDArray[np.complex128] | None = None,
        n_points: int = 20,
    ) -> list[dict]:
        """Sweep magic_strength and measure code properties."""
        from ..measures.magic import stabilizer_renyi_entropy, nonlocal_magic
        from ..measures.entanglement import subsystem_entropy

        if logical_state is None:
            logical_state = np.array([1, 0], dtype=np.complex128)

        results = []
        for eps in np.linspace(0, 2.0, n_points):
            self.config.magic_strength = eps
            state = self.encode(logical_state)

            # Entropy of first half of boundary
            n_half = self.config.n_physical // 2
            s_half = subsystem_entropy(state, list(range(n_half)))

            results.append({
                "magic_strength": eps,
                "sre": stabilizer_renyi_entropy(state),
                "nonlocal_magic": nonlocal_magic(state),
                "boundary_entropy_half": s_half,
            })

        return results
