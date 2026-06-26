"""Tests for quantum gates."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqhg.core.gates import (
    pauli_x, pauli_y, pauli_z, hadamard, identity_2,
    s_gate, t_gate, cz, cnot, ccz, toffoli,
    controlled_phase, multi_controlled_phase, pauli_tensor, rz,
)


def assert_unitary(U, atol=1e-12):
    """Check U†U = I."""
    n = U.shape[0]
    product = U.conj().T @ U
    np.testing.assert_allclose(product, np.eye(n), atol=atol)


class TestSingleQubitGates:
    def test_paulis_hermitian(self):
        for P in [pauli_x, pauli_y, pauli_z]:
            np.testing.assert_allclose(P, P.conj().T, atol=1e-15)

    def test_paulis_square_to_identity(self):
        for P in [pauli_x, pauli_y, pauli_z]:
            np.testing.assert_allclose(P @ P, identity_2, atol=1e-15)

    def test_hadamard_unitary(self):
        assert_unitary(hadamard)

    def test_hadamard_involutory(self):
        np.testing.assert_allclose(hadamard @ hadamard, identity_2, atol=1e-15)

    def test_s_gate_unitary(self):
        assert_unitary(s_gate)

    def test_t_gate_unitary(self):
        assert_unitary(t_gate)

    def test_t_squared_is_s(self):
        np.testing.assert_allclose(t_gate @ t_gate, s_gate, atol=1e-15)

    def test_rz_at_zero(self):
        np.testing.assert_allclose(rz(0), identity_2, atol=1e-15)


class TestTwoQubitGates:
    def test_cz_unitary(self):
        assert_unitary(cz)

    def test_cz_symmetric(self):
        np.testing.assert_allclose(cz, cz.T, atol=1e-15)

    def test_cnot_unitary(self):
        assert_unitary(cnot)

    def test_controlled_phase_pi_is_cz(self):
        cp = controlled_phase(np.pi)
        np.testing.assert_allclose(cp, cz, atol=1e-15)


class TestThreeQubitGates:
    def test_ccz_unitary(self):
        assert_unitary(ccz)

    def test_ccz_diagonal(self):
        assert np.allclose(ccz, np.diag(np.diag(ccz)))

    def test_ccz_only_flips_111(self):
        diag = np.diag(ccz)
        expected = np.array([1, 1, 1, 1, 1, 1, 1, -1])
        np.testing.assert_allclose(diag, expected)

    def test_toffoli_unitary(self):
        assert_unitary(toffoli)

    def test_multi_controlled_phase(self):
        # 3-qubit version with pi should match CCZ
        mcp = multi_controlled_phase(3, np.pi)
        np.testing.assert_allclose(mcp, ccz, atol=1e-15)


class TestPauliTensor:
    def test_single_paulis(self):
        np.testing.assert_allclose(pauli_tensor("X"), pauli_x)
        np.testing.assert_allclose(pauli_tensor("Z"), pauli_z)
        np.testing.assert_allclose(pauli_tensor("I"), identity_2)

    def test_two_qubit(self):
        xz = pauli_tensor("XZ")
        expected = np.kron(pauli_x, pauli_z)
        np.testing.assert_allclose(xz, expected)

    def test_identity_string(self):
        ii = pauli_tensor("II")
        np.testing.assert_allclose(ii, np.eye(4))
