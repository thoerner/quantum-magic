"""Fast classical shadow simulation and estimation (no Qiskit dependency).

Operates directly on ``mqhg.core.statevector.Statevector`` and stores shadows
in a vectorized array form for efficient SRE and mutual-information estimation
at large system sizes (n up to ~20, where the statevector still fits in RAM).

This complements ``mqhg.hardware.shadows`` (which targets real Qiskit hardware)
by providing a pure-NumPy path for scaling studies and pipeline validation.

Basis convention (matches ShadowProtocol):
    X basis -> apply H before Z-measurement
    Y basis -> apply S^dagger then H
    Z basis -> no rotation
Basis codes: 0 = X, 1 = Y, 2 = Z.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector


_H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)
# Rotation that maps the measured-basis eigenstates to the computational basis.
_ROT = {
    0: _H,            # X
    1: _H @ _SDG,     # Y
    2: np.eye(2, dtype=np.complex128),  # Z
}


@dataclass
class VectorizedShadows:
    """Classical shadows stored as dense integer arrays.

    Attributes:
        n_qubits: Number of qubits.
        bases: int8 array of shape (n_snapshots, n_qubits); 0=X, 1=Y, 2=Z.
        outcomes: int8 array of shape (n_snapshots, n_qubits); 0/1 measurement.
    """

    n_qubits: int
    bases: NDArray[np.int8]
    outcomes: NDArray[np.int8]

    @property
    def n_snapshots(self) -> int:
        return self.bases.shape[0]

    def _pauli_contributions(self, pauli: NDArray[np.int8]) -> NDArray[np.float64] | None:
        """Per-snapshot single-shot shadow estimates for a Pauli string.

        Codes: -1 = identity, 0 = X, 1 = Y, 2 = Z. Returns None for the
        identity (handled separately). Each entry is the unbiased single-shot
        estimate (3 per matched support qubit, 0 if any support qubit's basis
        mismatches); the plain mean is an unbiased estimate of <P>.
        """
        support = pauli >= 0
        if not np.any(support):
            return None
        sup_idx = np.where(support)[0]
        sup_pauli = pauli[sup_idx]
        match = np.all(self.bases[:, sup_idx] == sup_pauli[None, :], axis=1)
        signs = (-1.0) ** self.outcomes[:, sup_idx]
        contrib = np.prod(signs, axis=1) * (3.0 ** len(sup_idx))
        return np.where(match, contrib, 0.0)

    def estimate_pauli(self, pauli: NDArray[np.int8]) -> float:
        """Estimate <P> (unbiased mean over all snapshots).

        Codes: -1 = identity, 0 = X, 1 = Y, 2 = Z.
        """
        contrib = self._pauli_contributions(pauli)
        if contrib is None:
            return 1.0  # identity
        return float(np.mean(contrib))

    def estimate_pauli_power_unbiased(
        self, pauli: NDArray[np.int8], power: int, group_perm: NDArray[np.int64]
    ) -> float:
        """Unbiased U-statistic estimate of |<P>|^power via independent groups.

        Splits the (permuted) snapshots into ``power`` disjoint groups, estimates
        <P> within each group, and multiplies. Because the groups are independent,
        the product is an unbiased estimator of <P>^power, free of the variance
        self-bias that plagues the naive (mean)^power estimator.

        Args:
            pauli: Pauli code array.
            power: Exponent (e.g. 2 or 4).
            group_perm: A permutation of snapshot indices (shared across Paulis
                for a given SRE estimate so groups are consistent).
        """
        contrib = self._pauli_contributions(pauli)
        if contrib is None:
            return 1.0  # identity to any power
        m = len(contrib)
        g = m // power
        if g == 0:
            return float(np.mean(contrib)) ** power
        prod = 1.0
        for t in range(power):
            idx = group_perm[t * g:(t + 1) * g]
            prod *= float(np.mean(contrib[idx]))
        return prod

    def estimate_two_qubit_rdm(self, i: int, j: int) -> NDArray[np.complex128]:
        """Estimate the 2-qubit reduced density matrix rho_ij from shadows.

        rho_ij = (1/4) sum_{P,Q in {I,X,Y,Z}} <P_i Q_j> (P ⊗ Q).
        The result is Hermitized; eigenvalues are clamped to >= 0 and renormalized
        by the caller (see _entropy_from_rdm).
        """
        paulis = [
            np.eye(2, dtype=np.complex128),
            np.array([[0, 1], [1, 0]], dtype=np.complex128),       # X
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),    # Y
            np.array([[1, 0], [0, -1]], dtype=np.complex128),      # Z
        ]
        # code: index 0 -> identity(-1), 1->X(0), 2->Y(1), 3->Z(2)
        code_map = {0: -1, 1: 0, 2: 1, 3: 2}

        rho = np.zeros((4, 4), dtype=np.complex128)
        for a in range(4):
            for b in range(4):
                if a == 0 and b == 0:
                    exp = 1.0
                else:
                    pauli = np.full(self.n_qubits, -1, dtype=np.int8)
                    if a != 0:
                        pauli[i] = code_map[a]
                    if b != 0:
                        pauli[j] = code_map[b]
                    exp = self.estimate_pauli(pauli)
                rho += exp * np.kron(paulis[a], paulis[b])
        rho /= 4.0
        return 0.5 * (rho + rho.conj().T)

    def estimate_single_qubit_rdm(self, i: int) -> NDArray[np.complex128]:
        """Estimate the single-qubit reduced density matrix rho_i."""
        paulis = [
            np.eye(2, dtype=np.complex128),
            np.array([[0, 1], [1, 0]], dtype=np.complex128),
            np.array([[0, -1j], [1j, 0]], dtype=np.complex128),
            np.array([[1, 0], [0, -1]], dtype=np.complex128),
        ]
        code_map = {0: -1, 1: 0, 2: 1, 3: 2}
        rho = np.eye(2, dtype=np.complex128)  # identity term <I>=1
        for a in range(1, 4):
            pauli = np.full(self.n_qubits, -1, dtype=np.int8)
            pauli[i] = code_map[a]
            rho += self.estimate_pauli(pauli) * paulis[a]
        rho /= 2.0
        return 0.5 * (rho + rho.conj().T)

    def _single_exp_on(self, i: int, p: int, idx: NDArray[np.int64]) -> float:
        """<P_i> estimated on a snapshot subset (p in {0,1,2} = X,Y,Z)."""
        match = self.bases[idx, i] == p
        signs = (-1.0) ** self.outcomes[idx, i]
        return float(np.mean(np.where(match, 3.0 * signs, 0.0)))

    def _pair_exp_on(self, i: int, j: int, p: int, q: int, idx: NDArray[np.int64]) -> float:
        """<P_i Q_j> estimated on a snapshot subset."""
        match = (self.bases[idx, i] == p) & (self.bases[idx, j] == q)
        signs = (-1.0) ** (self.outcomes[idx, i] + self.outcomes[idx, j])
        return float(np.mean(np.where(match, 9.0 * signs, 0.0)))

    def estimate_correlator_backreaction(self, seed: int | None = None) -> float:
        """Shadow estimate of the connected-correlator backreaction ||G||_F.

        G[i,j] = sqrt( sum_{P,Q in {X,Y,Z}} |<P_i Q_j> - <P_i><Q_j>|^2 ),
        the connected two-point correlation strength. Only weight-<=2 Paulis
        are involved, which classical shadows estimate efficiently (unlike SRE).

        Bias control: the snapshots are split into two independent halves; the
        squared connected correlator is estimated as the cross-half product
        c_A * c_B, removing the dominant variance self-bias.
        """
        rng = np.random.default_rng(seed)
        n = self.n_qubits
        perm = rng.permutation(self.n_snapshots)
        half = len(perm) // 2
        A, B = perm[:half], perm[half:]

        g2 = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                sA = {p: self._single_exp_on(i, p, A) for p in range(3)}
                sjA = {q: self._single_exp_on(j, q, A) for q in range(3)}
                sB = {p: self._single_exp_on(i, p, B) for p in range(3)}
                sjB = {q: self._single_exp_on(j, q, B) for q in range(3)}
                acc = 0.0
                for p in range(3):
                    for q in range(3):
                        cA = self._pair_exp_on(i, j, p, q, A) - sA[p] * sjA[q]
                        cB = self._pair_exp_on(i, j, p, q, B) - sB[p] * sjB[q]
                        acc += cA * cB  # unbiased estimate of |c|^2
                g2[i, j] = g2[j, i] = max(0.0, acc)
        return float(np.sqrt(np.sum(g2)))

    def estimate_mi_matrix(self) -> NDArray[np.float64]:
        """Estimate the pairwise mutual information matrix from shadows.

        I(i:j) = S(rho_i) + S(rho_j) - S(rho_ij), using shadow-estimated RDMs
        with eigenvalue clamping. This is the shadow analogue of
        mqhg.measures.entanglement.mutual_information_matrix and avoids the
        O(n^2 2^n) exact partial traces.
        """
        n = self.n_qubits
        s_single = np.array([_entropy_from_rdm(self.estimate_single_qubit_rdm(i))
                             for i in range(n)])
        mi = np.zeros((n, n), dtype=np.float64)
        for i in range(n):
            for j in range(i + 1, n):
                s_ij = _entropy_from_rdm(self.estimate_two_qubit_rdm(i, j))
                val = max(0.0, s_single[i] + s_single[j] - s_ij)
                mi[i, j] = val
                mi[j, i] = val
        return mi


def _entropy_from_rdm(rho: NDArray[np.complex128]) -> float:
    """Von Neumann entropy of a (possibly non-PSD) shadow-estimated RDM.

    Eigenvalues are clamped to >= 0 and renormalized to sum to 1.
    """
    eig = np.linalg.eigvalsh(rho)
    eig = np.clip(np.real(eig), 0.0, None)
    total = eig.sum()
    if total < 1e-12:
        return 0.0
    eig = eig / total
    eig = eig[eig > 1e-12]
    return float(-np.sum(eig * np.log2(eig)))


def simulate_shadows_fast(
    state: Statevector,
    n_measurements: int = 1000,
    seed: int | None = None,
    shots_per_basis: int = 1,
) -> VectorizedShadows:
    """Simulate classical shadows by direct statevector sampling (no Qiskit).

    For each measurement round a random Pauli basis is drawn per qubit, the
    state is rotated into that basis, and an outcome is sampled from the Born
    distribution.

    Args:
        state: Target state (our own Statevector).
        n_measurements: Number of random measurement bases.
        seed: RNG seed.
        shots_per_basis: Outcomes sampled per rotated state. Using >1 amortizes
            the O(n 2^n) rotation cost across several snapshots; snapshots within
            a basis group share the same basis (a mild correlation that is
            acceptable for SRE/MI estimation when n_measurements is large).

    Returns:
        VectorizedShadows with n_measurements * shots_per_basis snapshots.
    """
    rng = np.random.default_rng(seed)
    n = state.n_qubits
    dim = 1 << n

    total = n_measurements * shots_per_basis
    bases = np.empty((total, n), dtype=np.int8)
    outcomes = np.empty((total, n), dtype=np.int8)

    outcome_codes = np.arange(dim, dtype=np.int64)
    # Precompute bit decomposition of every basis-state index (dim, n).
    bit_table = ((outcome_codes[:, None] >> np.arange(n)[None, :]) & 1).astype(np.int8)

    row = 0
    for _ in range(n_measurements):
        basis = rng.integers(0, 3, size=n).astype(np.int8)
        rotated = state
        for q in range(n):
            rotated = rotated.apply_gate(_ROT[int(basis[q])], [q])
        probs = np.abs(rotated.amplitudes) ** 2
        probs = probs / probs.sum()
        sampled = rng.choice(dim, size=shots_per_basis, p=probs)
        for s in sampled:
            bases[row] = basis
            outcomes[row] = bit_table[s]
            row += 1

    return VectorizedShadows(n_qubits=n, bases=bases, outcomes=outcomes)


def estimate_sre_fast(
    shadows: VectorizedShadows,
    n_pauli_samples: int = 500,
    alpha: int = 2,
    seed: int | None = None,
) -> float:
    """Estimate SRE from vectorized shadows via uniform Pauli importance sampling.

    For alpha=2, ``M_2 = -log2( mean_P |<P>|^4 ) - n`` with P uniform over
    {I,X,Y,Z}^n. The inner ``|<P>|^4`` is estimated with an UNBIASED 4-group
    U-statistic (see VectorizedShadows.estimate_pauli_power_unbiased), which
    removes the variance self-bias that otherwise inflates the estimate and
    collapses the SRE to zero at larger n.
    """
    rng = np.random.default_rng(seed)
    n = shadows.n_qubits
    power = 2 * alpha  # |<P>|^{2 alpha}

    # One shared random permutation defines the independent groups for all Paulis.
    group_perm = rng.permutation(shadows.n_snapshots)

    xi_pow = np.empty(n_pauli_samples, dtype=np.float64)
    for k in range(n_pauli_samples):
        draw = rng.integers(0, 4, size=n)
        pauli = np.where(draw == 0, -1, draw - 1).astype(np.int8)
        xi_pow[k] = shadows.estimate_pauli_power_unbiased(pauli, power, group_perm)

    mean_pow = float(np.mean(xi_pow))
    if mean_pow < 1e-30:
        return float(n)
    if alpha == 2:
        return float(max(0.0, -np.log2(mean_pow) - n))
    return float(max(0.0, (1 / (1 - alpha)) * np.log2(mean_pow) - n))
