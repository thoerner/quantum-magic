"""Tests for entanglement measures."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqhg.core.statevector import Statevector
from mqhg.core.gates import hadamard, cnot, cz
from mqhg.measures.entanglement import (
    von_neumann_entropy, renyi_entropy, subsystem_entropy,
    mutual_information, mutual_information_matrix, entanglement_spectrum,
)


class TestVonNeumannEntropy:
    def test_pure_state_zero_entropy(self):
        rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        assert abs(von_neumann_entropy(rho)) < 1e-14

    def test_maximally_mixed_entropy(self):
        rho = np.eye(2, dtype=np.complex128) / 2
        assert abs(von_neumann_entropy(rho) - 1.0) < 1e-14

    def test_maximally_mixed_4d(self):
        rho = np.eye(4, dtype=np.complex128) / 4
        assert abs(von_neumann_entropy(rho) - 2.0) < 1e-14


class TestRenyiEntropy:
    def test_renyi_2_pure(self):
        rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        assert abs(renyi_entropy(rho, alpha=2.0)) < 1e-14

    def test_renyi_2_maximally_mixed(self):
        rho = np.eye(2, dtype=np.complex128) / 2
        assert abs(renyi_entropy(rho, alpha=2.0) - 1.0) < 1e-14

    def test_renyi_1_equals_von_neumann(self):
        rho = np.array([[0.7, 0.1], [0.1, 0.3]], dtype=np.complex128)
        assert abs(renyi_entropy(rho, alpha=1.0) - von_neumann_entropy(rho)) < 1e-10


class TestMutualInformation:
    def test_product_state_zero_mi(self):
        sv = Statevector.plus_n(4)
        mi = mutual_information(sv, [0], [2])
        assert abs(mi) < 1e-12

    def test_bell_state_mi(self):
        sv = Statevector.zero_n(2)
        sv = sv.apply_gate(hadamard, [0])
        sv = sv.apply_gate(cnot, [0, 1])
        mi = mutual_information(sv, [0], [1])
        # Bell state: I(A:B) = 2 (maximal for 2 qubits)
        assert abs(mi - 2.0) < 1e-12

    def test_mi_matrix_symmetric(self):
        sv = Statevector.plus_n(4)
        sv = sv.apply_gate(cz, [0, 1])
        sv = sv.apply_gate(cz, [2, 3])
        mi = mutual_information_matrix(sv)
        np.testing.assert_allclose(mi, mi.T, atol=1e-14)

    def test_mi_matrix_diagonal_zero(self):
        sv = Statevector.plus_n(4)
        sv = sv.apply_gate(cz, [0, 1])
        mi = mutual_information_matrix(sv)
        np.testing.assert_allclose(np.diag(mi), 0, atol=1e-14)


class TestEntanglementSpectrum:
    def test_product_state_spectrum(self):
        sv = Statevector.plus_n(4)
        spec = entanglement_spectrum(sv, [0, 1])
        # Product state: single eigenvalue = 1
        assert abs(spec[0] - 1.0) < 1e-12
        assert np.sum(spec[1:]) < 1e-12

    def test_bell_state_spectrum(self):
        sv = Statevector.zero_n(2)
        sv = sv.apply_gate(hadamard, [0])
        sv = sv.apply_gate(cnot, [0, 1])
        spec = entanglement_spectrum(sv, [0])
        np.testing.assert_allclose(spec, [0.5, 0.5], atol=1e-14)
