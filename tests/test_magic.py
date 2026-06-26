"""Tests for magic (non-stabilizerness) measures."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqhg.core.statevector import Statevector
from mqhg.core.gates import hadamard, t_gate, cz
from mqhg.states.graph import GraphState
from mqhg.states.hypergraph import Hypergraph, HypergraphState
from mqhg.measures.magic import (
    stabilizer_renyi_entropy, nonlocal_magic, pauli_expectation_distribution,
)


class TestStabilizerRenyiEntropy:
    def test_computational_basis_is_stabilizer(self):
        """Computational basis states are stabilizer states → SRE = 0."""
        sv = Statevector.zero_n(2)
        sre = stabilizer_renyi_entropy(sv)
        assert sre < 1e-10

    def test_plus_state_is_stabilizer(self):
        """|+>^n is a stabilizer state → SRE = 0."""
        sv = Statevector.plus_n(3)
        sre = stabilizer_renyi_entropy(sv)
        assert sre < 1e-10

    def test_graph_state_is_stabilizer(self):
        """Graph states are stabilizer states → SRE = 0."""
        gs = GraphState.ring(4)
        state = gs.prepare()
        sre = stabilizer_renyi_entropy(state)
        assert sre < 1e-10

    def test_t_state_has_magic(self):
        """T|+> has nonzero magic."""
        sv = Statevector.plus_n(1)
        sv = sv.apply_gate(t_gate, [0])
        sre = stabilizer_renyi_entropy(sv)
        assert sre > 0.01

    def test_ccz_hypergraph_has_magic(self):
        """Hypergraph state with CCZ edges should have magic."""
        hg = Hypergraph(4)
        hg.add_edge((0, 1, 2))
        hg.add_edge((1, 2, 3))
        state = HypergraphState(hg).prepare()
        sre = stabilizer_renyi_entropy(state)
        assert sre > 0.01


class TestNonlocalMagic:
    def test_product_state_zero(self):
        """Product states should have zero non-local magic."""
        sv = Statevector.plus_n(3)
        nl = nonlocal_magic(sv)
        assert nl < 1e-10

    def test_graph_state_zero(self):
        """Graph states are stabilizer → zero magic (local and non-local)."""
        gs = GraphState.ring(4)
        state = gs.prepare()
        nl = nonlocal_magic(state)
        assert nl < 1e-10

    def test_nonlocal_magic_nonnegative(self):
        """Non-local magic should be non-negative."""
        hg = Hypergraph(4)
        hg.add_edge((0, 1, 2))
        state = HypergraphState(hg).prepare()
        nl = nonlocal_magic(state)
        assert nl >= -1e-10


class TestPauliDistribution:
    def test_stabilizer_state_sparse(self):
        """Stabilizer states have sparse Pauli distribution (many zeros, some ±1)."""
        sv = Statevector.zero_n(2)
        dist = pauli_expectation_distribution(sv)
        # For |00>, <P> is ±1 for Paulis in stabilizer group, 0 otherwise
        n_nonzero = np.sum(dist > 0.5)
        assert n_nonzero == 4  # |stabilizer group| = 2^n = 4 for n=2

    def test_t_state_less_sparse(self):
        """T-state should have a less sparse Pauli distribution."""
        sv = Statevector.plus_n(2)
        sv = sv.apply_gate(t_gate, [0])
        sv = sv.apply_gate(t_gate, [1])
        dist = pauli_expectation_distribution(sv)
        n_nonzero = np.sum(dist > 0.01)
        # Should have more nonzero entries than a stabilizer state
        assert n_nonzero > 4
