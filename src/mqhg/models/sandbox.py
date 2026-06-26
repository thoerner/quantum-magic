"""Model 1: Hypergraph-state sandbox.

The parametric hypergraph state:
    |ψ_H(θ)> = ∏_{e ∈ E} exp(iθ_e ∏_{v∈e} Z_v) |+>^⊗n

Varies edge size (2-body, 3-body, 4-body), edge density, and phase angles
to probe the relationship between magic and geometry deformation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector
from ..states.hypergraph import Hypergraph, HypergraphState
from ..measures.entanglement import mutual_information_matrix
from ..measures.magic import stabilizer_renyi_entropy, nonlocal_magic
from ..measures.geometry import mutual_info_distance_matrix


@dataclass
class SandboxConfig:
    """Configuration for a sandbox experiment."""

    n_qubits: int = 8
    edges_2body: list[tuple[int, int]] = field(default_factory=list)
    edges_3body: list[tuple[int, int, int]] = field(default_factory=list)
    edges_4body: list[tuple[int, int, int, int]] = field(default_factory=list)
    phase_2body: float = np.pi
    phase_3body: float = np.pi
    phase_4body: float = np.pi


@dataclass
class SandboxResult:
    """Results from a sandbox run."""

    state: Statevector
    sre: float
    nonlocal_magic: float
    mi_matrix: NDArray[np.float64]
    distance_matrix: NDArray[np.float64]
    backreaction: float


class HypergraphSandbox:
    """Model 1: Systematically explore hypergraph states and their geometry."""

    def __init__(self, config: SandboxConfig):
        self.config = config

    def build_hypergraph(self) -> Hypergraph:
        cfg = self.config
        hg = Hypergraph(cfg.n_qubits)

        for e in cfg.edges_2body:
            hg.add_edge(e, cfg.phase_2body)
        for e in cfg.edges_3body:
            hg.add_edge(e, cfg.phase_3body)
        for e in cfg.edges_4body:
            hg.add_edge(e, cfg.phase_4body)

        return hg

    def prepare_state(self) -> Statevector:
        hg = self.build_hypergraph()
        return HypergraphState(hg).prepare()

    def run(self) -> SandboxResult:
        """Prepare state and compute all observables."""
        state = self.prepare_state()
        mi = mutual_information_matrix(state)
        dist = mutual_info_distance_matrix(state)
        sre = stabilizer_renyi_entropy(state)
        nl_magic = nonlocal_magic(state)

        return SandboxResult(
            state=state,
            sre=sre,
            nonlocal_magic=nl_magic,
            mi_matrix=mi,
            distance_matrix=dist,
            backreaction=0.0,  # Computed separately with perturbation
        )

    def measure_backreaction(self, reference_mi: NDArray[np.float64] | None = None) -> float:
        """Measure geometric deformation: ||MI(ψ) - MI_ref||_F.

        Backreaction = how much MI structure (emergent geometry) the state
        has compared to the stabilizer reference (which has zero pairwise MI
        for graph states). Magic creates geometry from flatness.

        Args:
            reference_mi: MI matrix of the reference state. If None, uses zero
                         (appropriate for graph state references with no pairwise MI).
        """
        state = self.prepare_state()
        mi = mutual_information_matrix(state)
        if reference_mi is None:
            reference_mi = np.zeros_like(mi)
        return float(np.linalg.norm(mi - reference_mi, "fro"))

    @classmethod
    def sweep_magic(
        cls,
        n_qubits: int,
        base_edges: list[tuple[int, int]],
        magic_edges: list[tuple[int, ...]],
        n_phases: int = 20,
    ) -> list[dict]:
        """Sweep non-Clifford phase from 0 to π and measure observables.

        Returns list of dicts with phase, sre, nonlocal_magic, backreaction.
        """
        results = []
        phases = np.linspace(0, np.pi, n_phases)

        for phi in phases:
            cfg = SandboxConfig(
                n_qubits=n_qubits,
                edges_2body=base_edges,
                edges_3body=[e for e in magic_edges if len(e) == 3],
                edges_4body=[e for e in magic_edges if len(e) == 4],
                phase_2body=np.pi,
                phase_3body=phi,
                phase_4body=phi,
            )
            sandbox = cls(cfg)
            result = sandbox.run()
            br = sandbox.measure_backreaction()

            results.append({
                "phase": float(phi),
                "sre": result.sre,
                "nonlocal_magic": result.nonlocal_magic,
                "backreaction": br,
            })

        return results
