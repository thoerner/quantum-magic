"""Tests for state preparation."""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mqhg.core.statevector import Statevector
from mqhg.states.graph import GraphState
from mqhg.states.hypergraph import Hypergraph, HypergraphState
from mqhg.states.random import random_clifford_state, random_magic_state, haar_random_state


class TestStatevector:
    def test_plus_n_normalized(self):
        for n in range(1, 6):
            sv = Statevector.plus_n(n)
            assert abs(sv.norm() - 1.0) < 1e-14

    def test_plus_n_uniform(self):
        sv = Statevector.plus_n(3)
        probs = sv.probabilities()
        np.testing.assert_allclose(probs, np.ones(8) / 8, atol=1e-14)

    def test_zero_n(self):
        sv = Statevector.zero_n(3)
        assert sv.amplitudes[0] == 1.0
        assert np.sum(np.abs(sv.amplitudes[1:])) < 1e-15

    def test_apply_gate_preserves_norm(self):
        from mqhg.core.gates import hadamard, cz
        sv = Statevector.zero_n(3)
        sv = sv.apply_gate(hadamard, [0])
        assert abs(sv.norm() - 1.0) < 1e-14
        sv = sv.apply_gate(cz, [0, 1])
        assert abs(sv.norm() - 1.0) < 1e-14

    def test_reduced_density_matrix_pure(self):
        sv = Statevector.zero_n(2)
        rho = sv.reduced_density_matrix([0])
        expected = np.array([[1, 0], [0, 0]], dtype=np.complex128)
        np.testing.assert_allclose(rho, expected, atol=1e-14)

    def test_reduced_density_matrix_bell_state(self):
        from mqhg.core.gates import hadamard, cnot
        sv = Statevector.zero_n(2)
        sv = sv.apply_gate(hadamard, [0])
        sv = sv.apply_gate(cnot, [0, 1])
        rho = sv.reduced_density_matrix([0])
        # Should be maximally mixed
        np.testing.assert_allclose(rho, np.eye(2) / 2, atol=1e-14)


class TestGraphState:
    def test_graph_state_normalized(self):
        gs = GraphState.ring(5)
        state = gs.prepare()
        assert abs(state.norm() - 1.0) < 1e-14

    def test_linear_graph_state(self):
        gs = GraphState.linear(4)
        state = gs.prepare()
        assert state.n_qubits == 4
        assert abs(state.norm() - 1.0) < 1e-14

    def test_graph_state_is_real(self):
        """Graph states from CZ on |+> should have real amplitudes."""
        gs = GraphState.ring(4)
        state = gs.prepare()
        np.testing.assert_allclose(state.amplitudes.imag, 0, atol=1e-14)


class TestHypergraphState:
    def test_hypergraph_normalized(self):
        hg = Hypergraph(4)
        hg.add_edge((0, 1, 2))
        hg.add_edge((1, 2, 3))
        state = HypergraphState(hg).prepare()
        assert abs(state.norm() - 1.0) < 1e-14

    def test_2body_hypergraph_equals_graph_state(self):
        """Hypergraph with only 2-body edges = graph state."""
        n = 4
        edges = [(0, 1), (1, 2), (2, 3)]

        # Hypergraph version
        hg = Hypergraph(n)
        hg.add_edges(edges)
        state_hg = HypergraphState(hg).prepare()

        # Graph state version
        import networkx as nx
        G = nx.Graph()
        G.add_nodes_from(range(n))
        G.add_edges_from(edges)
        state_gs = GraphState(G).prepare()

        np.testing.assert_allclose(
            state_hg.amplitudes, state_gs.amplitudes, atol=1e-14
        )

    def test_random_hypergraph(self):
        rng = np.random.default_rng(42)
        hs = HypergraphState.random(6, n_edges=5, max_edge_size=3, rng=rng)
        state = hs.prepare()
        assert abs(state.norm() - 1.0) < 1e-14


class TestRandomStates:
    def test_clifford_state_normalized(self):
        state = random_clifford_state(5, depth=3, rng=np.random.default_rng(0))
        assert abs(state.norm() - 1.0) < 1e-14

    def test_magic_state_normalized(self):
        state = random_magic_state(5, depth=3, p_magic=0.3, rng=np.random.default_rng(0))
        assert abs(state.norm() - 1.0) < 1e-14

    def test_haar_random_normalized(self):
        state = haar_random_state(5, rng=np.random.default_rng(0))
        assert abs(state.norm() - 1.0) < 1e-14
