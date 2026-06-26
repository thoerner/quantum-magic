"""Phase 3: Small-system exact numerics with hypergraph states.

Implements the core experiment:
1. Construct Clifford graph states (no magic)
2. Construct hypergraph states with CCZ gates (magic)
3. Compare geometry deformation under localized excitations
4. Test Conjectures A, B, C from the research spec
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")

from mqhg.states.graph import GraphState
from mqhg.states.hypergraph import HypergraphState, Hypergraph
from mqhg.states.random import random_clifford_state, random_magic_state, haar_random_state
from mqhg.measures.entanglement import mutual_information_matrix, subsystem_entropy
from mqhg.measures.magic import stabilizer_renyi_entropy, nonlocal_magic
from mqhg.measures.geometry import mutual_info_distance_matrix
from mqhg.analysis.backreaction import BackreactionObservable
from mqhg.analysis.plotting import (
    plot_magic_vs_backreaction,
    plot_geometry_deformation,
    plot_sweep_results,
)


OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def experiment_conjecture_b(n: int = 6):
    """Conjecture B: Stabilizer geometry is too rigid.

    Build stabilizer-only graph states, inject excitations,
    verify geometry does NOT deform significantly.
    """
    print(f"\n{'='*60}")
    print(f"CONJECTURE B TEST: Stabilizer rigidity (n={n})")
    print(f"{'='*60}")

    graphs = {
        "linear": GraphState.linear(n),
        "ring": GraphState.ring(n),
        "complete": GraphState.complete(n),
        "star": GraphState.star(n),
    }

    for name, gs in graphs.items():
        state = gs.prepare()
        sre = stabilizer_renyi_entropy(state)
        nl_magic = nonlocal_magic(state)

        br_obs = BackreactionObservable(state)
        br = br_obs.measure(0, "X")

        print(f"\n  {name} graph:")
        print(f"    SRE = {sre:.6f} (should be ~0 for stabilizer)")
        print(f"    Non-local magic = {nl_magic:.6f}")
        print(f"    Backreaction ||ΔD||_F = {br.distance_change_frobenius:.6f}")


def experiment_conjecture_c(n: int = 6):
    """Conjecture C: Magic enables backreaction.

    Add non-Clifford hyperedges and verify geometry DOES deform.
    """
    print(f"\n{'='*60}")
    print(f"CONJECTURE C TEST: Magic enables backreaction (n={n})")
    print(f"{'='*60}")

    # Stabilizer baseline: ring graph state
    ring = GraphState.ring(n)
    state_stab = ring.prepare()

    # Magic version: ring + 3-body hyperedges
    hg = Hypergraph(n)
    for i in range(n):
        hg.add_edge((i, (i + 1) % n))  # Ring edges (Clifford)
    for i in range(0, n - 2, 2):
        hg.add_edge((i, i + 1, i + 2))  # 3-body magic edges
    state_magic = HypergraphState(hg).prepare()

    # Compare
    for label, state in [("Stabilizer (ring)", state_stab), ("Magic (ring+CCZ)", state_magic)]:
        sre = stabilizer_renyi_entropy(state)
        nl = nonlocal_magic(state)
        br_obs = BackreactionObservable(state)
        br = br_obs.measure(0, "X")

        print(f"\n  {label}:")
        print(f"    SRE = {sre:.6f}")
        print(f"    Non-local magic = {nl:.6f}")
        print(f"    Backreaction ||ΔD||_F = {br.distance_change_frobenius:.6f}")
        print(f"    Backreaction (local) = {br.distance_change_local:.6f}")


def experiment_magic_sweep(n: int = 6):
    """Sweep non-Clifford phase from 0 to π and plot observables."""
    print(f"\n{'='*60}")
    print(f"MAGIC SWEEP: Phase 0→π (n={n})")
    print(f"{'='*60}")

    base_edges = [(i, (i + 1) % n) for i in range(n)]
    magic_edges = [(i, (i + 1) % n, (i + 2) % n) for i in range(0, n, 2)]

    from mqhg.models.sandbox import HypergraphSandbox
    results = HypergraphSandbox.sweep_magic(n, base_edges, magic_edges, n_phases=15)

    for r in results:
        print(f"  θ={r['phase']:.3f}: SRE={r['sre']:.4f}, "
              f"NL-magic={r['nonlocal_magic']:.4f}, BR={r['backreaction']:.4f}")

    # Plot
    plot_sweep_results(
        results,
        x_key="phase",
        y_keys=["sre", "nonlocal_magic", "backreaction"],
        title=f"Magic Sweep (n={n}): Phase → Observables",
        save_path=OUTPUT_DIR / "magic_sweep.png",
    )
    print(f"\n  Plot saved: {OUTPUT_DIR / 'magic_sweep.png'}")


def experiment_state_comparison(n: int = 6):
    """Compare all state classes on magic vs backreaction."""
    print(f"\n{'='*60}")
    print(f"STATE COMPARISON: All classes (n={n})")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)
    states = {}

    # Clifford states
    states["ring_graph"] = GraphState.ring(n).prepare()
    states["complete_graph"] = GraphState.complete(n).prepare()
    states["random_clifford"] = random_clifford_state(n, depth=5, rng=rng)

    # Magic states
    states["magic_p=0.1"] = random_magic_state(n, depth=5, p_magic=0.1, rng=rng)
    states["magic_p=0.3"] = random_magic_state(n, depth=5, p_magic=0.3, rng=rng)
    states["magic_p=0.7"] = random_magic_state(n, depth=5, p_magic=0.7, rng=rng)

    # Hypergraph states
    hg = Hypergraph(n)
    for i in range(n):
        hg.add_edge((i, (i + 1) % n))
    for i in range(0, n - 2):
        hg.add_edge((i, i + 1, i + 2))
    states["hypergraph_3body"] = HypergraphState(hg).prepare()

    # Haar random
    states["haar_random"] = haar_random_state(n, rng=rng)

    # Compute observables
    magic_vals = []
    br_vals = []
    labels = []

    for name, state in states.items():
        sre = stabilizer_renyi_entropy(state)
        nl = nonlocal_magic(state)
        br_obs = BackreactionObservable(state)
        br = br_obs.measure(0, "X")

        magic_vals.append(nl)
        br_vals.append(br.distance_change_frobenius)
        labels.append(name)

        print(f"  {name:20s}: SRE={sre:.4f}, NL-magic={nl:.4f}, BR={br.distance_change_frobenius:.4f}")

    # Plot magic vs backreaction
    plot_magic_vs_backreaction(
        magic_vals, br_vals, labels=labels,
        title=f"Non-local Magic vs Backreaction (n={n})",
        save_path=OUTPUT_DIR / "magic_vs_backreaction.png",
    )
    print(f"\n  Plot saved: {OUTPUT_DIR / 'magic_vs_backreaction.png'}")


def experiment_falsification_test1(n: int = 6):
    """Falsification Test 1: High entanglement, low magic.

    If high-entanglement-low-magic states backreact just as well,
    then magic is not necessary.
    """
    print(f"\n{'='*60}")
    print(f"FALSIFICATION TEST 1: Entanglement-only control (n={n})")
    print(f"{'='*60}")

    rng = np.random.default_rng(42)

    # High entanglement, low magic: deep Clifford circuit
    state_high_ent = random_clifford_state(n, depth=20, rng=rng)
    # High entanglement, high magic
    state_high_both = random_magic_state(n, depth=20, p_magic=0.5, rng=rng)

    for label, state in [("High-ent, low-magic (Clifford)", state_high_ent),
                         ("High-ent, high-magic", state_high_both)]:
        sre = stabilizer_renyi_entropy(state)
        nl = nonlocal_magic(state)
        mean_ent = np.mean([subsystem_entropy(state, [q]) for q in range(n)])
        br = BackreactionObservable(state).measure(0, "X")

        print(f"\n  {label}:")
        print(f"    Mean entropy = {mean_ent:.4f}")
        print(f"    SRE = {sre:.4f}")
        print(f"    Non-local magic = {nl:.4f}")
        print(f"    Backreaction = {br.distance_change_frobenius:.4f}")


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    experiment_conjecture_b(n)
    experiment_conjecture_c(n)
    experiment_magic_sweep(n)
    experiment_state_comparison(n)
    experiment_falsification_test1(n)

    print(f"\n\nAll Phase 3 experiments complete. Output in {OUTPUT_DIR}")
