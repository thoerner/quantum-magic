"""Quantum hypergraph state construction.

A hypergraph state is:
    |ψ_H> = ∏_{e ∈ E} U_e |+>^⊗n

where U_e is a controlled-phase gate acting on the vertices in hyperedge e.
Two-body edges (CZ) produce stabilizer graph states.
Three-body and higher edges (CCZ, etc.) inject magic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.statevector import Statevector
from ..core.gates import cz, ccz, multi_controlled_phase


@dataclass
class Hypergraph:
    """A weighted hypergraph defining a quantum state.

    Attributes:
        n_qubits: Number of vertices/qubits.
        edges: List of hyperedges, each a tuple of qubit indices.
        phases: Phase angle for each hyperedge (pi = standard CZ/CCZ).
    """

    n_qubits: int
    edges: list[tuple[int, ...]] = field(default_factory=list)
    phases: list[float] = field(default_factory=list)

    def add_edge(self, vertices: tuple[int, ...], phase: float = np.pi) -> None:
        """Add a hyperedge with specified phase."""
        for v in vertices:
            if v < 0 or v >= self.n_qubits:
                raise ValueError(f"Vertex {v} out of range [0, {self.n_qubits})")
        self.edges.append(vertices)
        self.phases.append(phase)

    def add_edges(self, edges: list[tuple[int, ...]], phase: float = np.pi) -> None:
        for e in edges:
            self.add_edge(e, phase)

    @property
    def max_edge_size(self) -> int:
        if not self.edges:
            return 0
        return max(len(e) for e in self.edges)

    @property
    def edge_size_distribution(self) -> dict[int, int]:
        dist: dict[int, int] = {}
        for e in self.edges:
            k = len(e)
            dist[k] = dist.get(k, 0) + 1
        return dist

    def is_graph(self) -> bool:
        """True if all edges have exactly 2 vertices (ordinary graph state)."""
        return all(len(e) == 2 for e in self.edges)


class HypergraphState:
    """Constructs and manipulates quantum hypergraph states."""

    def __init__(self, hypergraph: Hypergraph):
        self.hypergraph = hypergraph

    def prepare(self) -> Statevector:
        """Prepare |ψ_H> = ∏_e U_e |+>^⊗n."""
        n = self.hypergraph.n_qubits
        state = Statevector.plus_n(n)

        for edge, phase in zip(self.hypergraph.edges, self.hypergraph.phases):
            k = len(edge)
            if k < 2:
                continue

            if k == 2 and np.isclose(phase, np.pi):
                gate = cz
            elif k == 3 and np.isclose(phase, np.pi):
                gate = ccz
            else:
                gate = multi_controlled_phase(k, phase)

            state = state.apply_gate(gate, list(edge))

        return state

    @classmethod
    def from_edges(cls, n_qubits: int, edges: list[tuple[int, ...]], phase: float = np.pi) -> HypergraphState:
        """Convenience constructor from edge list."""
        hg = Hypergraph(n_qubits)
        hg.add_edges(edges, phase)
        return cls(hg)

    @classmethod
    def random(cls, n_qubits: int, n_edges: int, max_edge_size: int = 3,
               rng: np.random.Generator | None = None) -> HypergraphState:
        """Generate a random hypergraph state."""
        if rng is None:
            rng = np.random.default_rng()

        hg = Hypergraph(n_qubits)
        for _ in range(n_edges):
            k = rng.integers(2, max_edge_size + 1)
            vertices = tuple(sorted(rng.choice(n_qubits, size=k, replace=False)))
            hg.add_edge(vertices)

        return cls(hg)

    @classmethod
    def parametric(cls, n_qubits: int, edges: list[tuple[int, ...]], theta: float) -> HypergraphState:
        """Parametric hypergraph state with uniform phase theta on all edges.

        |ψ_H(θ)> = ∏_e exp(iθ ∏_{v∈e} Z_v) |+>^⊗n

        This is the Model 1 sandbox state from the research spec.
        """
        hg = Hypergraph(n_qubits)
        for e in edges:
            hg.add_edge(e, 2 * theta)  # controlled-phase(2θ) ≈ exp(iθ Z⊗Z⊗...)
        return cls(hg)
