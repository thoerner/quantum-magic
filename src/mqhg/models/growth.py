"""Model 4: Random circuit/hypergraph growth model.

Time-evolving quantum computational hypergraph:
    G_t = (V_t, E_t, W_t)
    |ψ_t> = U_t |ψ_0>
    U_t = product of local Clifford gates + sparse non-Clifford gates

The non-Clifford gate rate p_magic is the control parameter.
Goal: identify a phase transition or crossover:
    too little magic → rigid/non-gravitating geometry
    moderate magic → smooth deformable geometry
    too much magic/randomness → no clean geometry, just quantum chaos
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from ..core.statevector import Statevector
from ..core.gates import cz, ccz, t_gate, hadamard, s_gate
from ..measures.entanglement import mutual_information_matrix, subsystem_entropy
from ..measures.magic import stabilizer_renyi_entropy, nonlocal_magic
from ..measures.geometry import mutual_info_distance_matrix


_CLIFFORD_1Q = [hadamard, s_gate, hadamard @ s_gate, s_gate @ hadamard]


@dataclass
class GrowthConfig:
    """Configuration for random circuit growth."""

    n_qubits: int = 8
    depth: int = 10
    p_magic: float = 0.1
    seed: int = 42
    measure_every: int = 1


@dataclass
class GrowthSnapshot:
    """State of the system at a given circuit depth."""

    step: int
    sre: float
    nonlocal_magic: float
    mean_entropy: float
    geometry_smoothness: float


class RandomCircuitGrowth:
    """Model 4: Track geometry emergence as circuit depth and magic grow.

    Key observable: does there exist a p_magic regime where:
    - entanglement builds clean geometry (mutual-info distances are smooth)
    - magic enables that geometry to respond to perturbations
    - but not so much magic that geometry dissolves into noise
    """

    def __init__(self, config: GrowthConfig | None = None):
        self.config = config or GrowthConfig()
        self._rng = np.random.default_rng(self.config.seed)

    def run(self) -> list[GrowthSnapshot]:
        """Execute the random circuit and record snapshots."""
        cfg = self.config
        state = Statevector.plus_n(cfg.n_qubits)
        snapshots = []

        for step in range(cfg.depth):
            state = self._apply_layer(state)

            if step % cfg.measure_every == 0:
                snapshot = self._measure(state, step)
                snapshots.append(snapshot)

        return snapshots

    def _apply_layer(self, state: Statevector) -> Statevector:
        """Apply one layer of random gates."""
        cfg = self.config
        n = cfg.n_qubits

        # Single-qubit layer
        for q in range(n):
            if self._rng.random() < cfg.p_magic:
                state = state.apply_gate(t_gate, [q])
            else:
                gate = _CLIFFORD_1Q[self._rng.integers(len(_CLIFFORD_1Q))]
                state = state.apply_gate(gate, [q])

        # Two-qubit entangling layer (random CZ pairs)
        qubits = list(range(n))
        self._rng.shuffle(qubits)
        for i in range(0, n - 1, 2):
            state = state.apply_gate(cz, [qubits[i], qubits[i + 1]])

        # Optional CCZ for 3-body magic
        if n >= 3 and self._rng.random() < cfg.p_magic:
            targets = sorted(self._rng.choice(n, size=3, replace=False).tolist())
            state = state.apply_gate(ccz, targets)

        return state

    def _measure(self, state: Statevector, step: int) -> GrowthSnapshot:
        """Compute observables at current step."""
        sre = stabilizer_renyi_entropy(state)
        nl_magic = nonlocal_magic(state)

        # Mean single-qubit entropy (entanglement proxy)
        n = state.n_qubits
        entropies = [subsystem_entropy(state, [q]) for q in range(n)]
        mean_ent = float(np.mean(entropies))

        # Geometry smoothness: inverse coefficient of variation of MI distances
        dist = mutual_info_distance_matrix(state)
        upper = dist[np.triu_indices(n, k=1)]
        if np.std(upper) > 1e-10:
            smoothness = float(np.mean(upper) / np.std(upper))
        else:
            smoothness = float("inf")

        return GrowthSnapshot(
            step=step,
            sre=sre,
            nonlocal_magic=nl_magic,
            mean_entropy=mean_ent,
            geometry_smoothness=smoothness,
        )

    @classmethod
    def phase_diagram(
        cls,
        n_qubits: int = 6,
        depth: int = 8,
        p_magic_values: list[float] | None = None,
        n_seeds: int = 5,
    ) -> list[dict]:
        """Sweep p_magic to map the phase diagram.

        Returns averaged observables for each p_magic value.
        """
        if p_magic_values is None:
            p_magic_values = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]

        results = []
        for p in p_magic_values:
            sre_vals = []
            nl_magic_vals = []
            smoothness_vals = []
            entropy_vals = []

            for seed in range(n_seeds):
                cfg = GrowthConfig(
                    n_qubits=n_qubits,
                    depth=depth,
                    p_magic=p,
                    seed=seed,
                )
                model = cls(cfg)
                snapshots = model.run()
                final = snapshots[-1]

                sre_vals.append(final.sre)
                nl_magic_vals.append(final.nonlocal_magic)
                smoothness_vals.append(final.geometry_smoothness)
                entropy_vals.append(final.mean_entropy)

            results.append({
                "p_magic": p,
                "sre_mean": float(np.mean(sre_vals)),
                "sre_std": float(np.std(sre_vals)),
                "nonlocal_magic_mean": float(np.mean(nl_magic_vals)),
                "smoothness_mean": float(np.mean(smoothness_vals)),
                "smoothness_std": float(np.std(smoothness_vals)),
                "entropy_mean": float(np.mean(entropy_vals)),
            })

        return results
