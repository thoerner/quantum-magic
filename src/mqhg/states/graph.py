"""Stabilizer graph state construction.

A graph state is a hypergraph state where all edges have exactly 2 vertices
(CZ gates only). Graph states are stabilizer states with zero magic.
"""

from __future__ import annotations

import numpy as np
import networkx as nx

from ..core.statevector import Statevector
from ..core.gates import cz
from .hypergraph import Hypergraph, HypergraphState


class GraphState:
    """Stabilizer graph state from a NetworkX graph or adjacency list."""

    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.n_qubits = graph.number_of_nodes()

    def prepare(self) -> Statevector:
        """Prepare the graph state |G> = ∏_{(i,j)∈E} CZ_{ij} |+>^⊗n."""
        state = Statevector.plus_n(self.n_qubits)
        for i, j in self.graph.edges():
            state = state.apply_gate(cz, [i, j])
        return state

    @classmethod
    def linear(cls, n: int) -> GraphState:
        """Linear chain graph: 0-1-2-...-n."""
        return cls(nx.path_graph(n))

    @classmethod
    def ring(cls, n: int) -> GraphState:
        """Ring/cycle graph."""
        return cls(nx.cycle_graph(n))

    @classmethod
    def complete(cls, n: int) -> GraphState:
        """Complete graph K_n."""
        return cls(nx.complete_graph(n))

    @classmethod
    def star(cls, n: int) -> GraphState:
        """Star graph with n-1 leaves."""
        return cls(nx.star_graph(n - 1))

    @classmethod
    def random_regular(cls, n: int, degree: int, seed: int | None = None) -> GraphState:
        """Random regular graph."""
        g = nx.random_regular_graph(degree, n, seed=seed)
        return cls(g)

    @classmethod
    def lattice_2d(cls, rows: int, cols: int) -> GraphState:
        """2D grid/lattice graph."""
        return cls(nx.grid_2d_graph(rows, cols))

    def to_hypergraph(self) -> Hypergraph:
        """Convert to Hypergraph representation."""
        hg = Hypergraph(self.n_qubits)
        for i, j in self.graph.edges():
            hg.add_edge((i, j))
        return hg
