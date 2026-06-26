"""Falsification Tests 2–7 for Magic-Gravity Hypothesis.

Each test targets a specific failure mode of the conjecture:
2. Magic-only control: high magic without entanglement structure
3. Randomness confound: is it magic or just chaos?
4. Basis dependence: is the result frame-independent?
5. Locality recovery: does geometry remain approximately local?
6. Universality: do all magic sources curve geometry equally?
7. Scaling: does the effect persist with system size?

Hypergraph-focused extensions:
- Tests 5/6/7 scale to n=10-12 with FWHT-accelerated SRE
- Test 6 includes hyperedge-order sweep (k=3,4,5)
- Ollivier-Ricci curvature wired into tests 5 and 7
"""

import argparse
import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mqhg.core.statevector import Statevector
from mqhg.core.gates import t_gate, hadamard, s_gate, cz, ccz, phase_gate, multi_controlled_phase
from mqhg.states.graph import GraphState
from mqhg.states.hypergraph import HypergraphState, Hypergraph
from mqhg.states.random import random_clifford_state, random_magic_state, haar_random_state
from mqhg.measures.entanglement import mutual_information_matrix, subsystem_entropy
from mqhg.measures.magic import stabilizer_renyi_entropy, nonlocal_magic
from mqhg.measures.geometry import average_curvature, spectral_dimension
from mqhg.analysis.backreaction import BackreactionObservable
from mqhg.io.results import ExperimentResult


OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def falsification_test_2(n: int = 6):
    """Test 2: Magic-only control.

    Build states with high magic but weak or scrambled entanglement geometry.
    If they do NOT produce coherent geometry, then magic alone is insufficient
    (magic is necessary but not sufficient for gravity).

    Strategy: Apply T gates to a product state (no entanglement) and to a
    weakly-entangled state. Compare MI deformation to structured magic states.
    """
    print(f"\n{'='*60}")
    print(f"FALSIFICATION TEST 2: Magic-only control (n={n})")
    print(f"{'='*60}")
    print("  Question: Does magic WITHOUT entanglement structure create geometry?")

    rng = np.random.default_rng(42)
    results = []

    # Case A: Product state with T gates (high local magic, zero entanglement)
    state_product_magic = Statevector.plus_n(n)
    for q in range(n):
        for _ in range(3):
            state_product_magic = state_product_magic.apply_gate(t_gate, [q])

    # Case B: Weakly entangled + high magic (sparse CZ, many T)
    state_weak_ent_magic = Statevector.plus_n(n)
    for q in range(n):
        state_weak_ent_magic = state_weak_ent_magic.apply_gate(t_gate, [q])
    state_weak_ent_magic = state_weak_ent_magic.apply_gate(cz, [0, 1])

    # Case C: Structured magic (ring + CCZ, our positive control)
    hg = Hypergraph(n)
    for i in range(n):
        hg.add_edge((i, (i + 1) % n))
    for i in range(0, n - 2, 2):
        hg.add_edge((i, i + 1, i + 2))
    state_structured = HypergraphState(hg).prepare()

    # Case D: Haar random (high magic, high entanglement, no structure)
    state_haar = haar_random_state(n, rng=rng)

    cases = [
        ("Product + T gates (local magic only)", state_product_magic),
        ("Weak entanglement + T gates", state_weak_ent_magic),
        ("Ring + CCZ (structured magic)", state_structured),
        ("Haar random (unstructured magic)", state_haar),
    ]

    for label, state in cases:
        sre = stabilizer_renyi_entropy(state)
        nl = nonlocal_magic(state)
        br = BackreactionObservable(state).measure()
        mean_ent = np.mean([subsystem_entropy(state, [q]) for q in range(n)])

        results.append({
            "label": label, "sre": sre, "nonlocal_magic": nl,
            "mi_deformation": br.mi_deformation, "mean_entropy": mean_ent,
        })
        print(f"\n  {label}:")
        print(f"    SRE = {sre:.4f}, Non-local magic = {nl:.4f}")
        print(f"    Mean entropy = {mean_ent:.4f}")
        print(f"    MI deformation = {br.mi_deformation:.4f}")

    # Verdict
    structured_br = results[2]["mi_deformation"]
    local_only_br = results[0]["mi_deformation"]
    print(f"\n  VERDICT: Structured magic MI-deform = {structured_br:.4f}, "
          f"local-only = {local_only_br:.4f}")
    if local_only_br < 0.1 * structured_br:
        print("  → Magic alone is INSUFFICIENT. Entanglement structure required. "
              "Hypothesis SURVIVES.")
    else:
        print("  → Magic alone creates comparable geometry. Hypothesis may need revision.")

    return results


def falsification_test_3(n: int = 6):
    """Test 3: Randomness confound.

    High magic may simply mean random chaotic complexity. Check whether
    geometry smoothness degrades with increasing magic (suggesting chaos
    destroys coherent geometry) vs remains structured.

    Strategy: Compare the MI matrix structure (is it smooth/decaying or noisy?)
    for states at various magic levels. Compute a "geometry coherence" metric.
    """
    print(f"\n{'='*60}")
    print(f"FALSIFICATION TEST 3: Randomness confound (n={n})")
    print(f"{'='*60}")
    print("  Question: Does high magic destroy geometry coherence (= just chaos)?")

    rng = np.random.default_rng(42)

    # Build states with increasing magic from structured source
    base_edges = [(i, (i + 1) % n) for i in range(n)]
    magic_edges = [(i, (i + 1) % n, (i + 2) % n) for i in range(0, n, 2)]

    phases = np.linspace(0, np.pi, 10)
    results = []

    for phi in phases:
        hg = Hypergraph(n)
        for e in base_edges:
            hg.add_edge(e, np.pi)
        for e in magic_edges:
            hg.add_edge(e, phi)
        state = HypergraphState(hg).prepare()

        mi = mutual_information_matrix(state)
        sre = stabilizer_renyi_entropy(state)

        # Geometry coherence: ratio of largest to mean MI (structured = high ratio)
        upper_tri = mi[np.triu_indices(n, k=1)]
        mi_max = float(upper_tri.max()) if upper_tri.max() > 1e-10 else 0.0
        mi_mean = float(upper_tri.mean()) if upper_tri.mean() > 1e-10 else 0.0
        coherence = mi_max / mi_mean if mi_mean > 1e-10 else 0.0

        # Smoothness: variance of MI values (low variance = uniform/smooth)
        mi_variance = float(upper_tri.var())

        results.append({
            "phase": float(phi), "sre": sre,
            "mi_max": mi_max, "mi_mean": mi_mean,
            "coherence": coherence, "variance": mi_variance,
        })

    # Also test Haar random (maximum chaos)
    haar = haar_random_state(n, rng=rng)
    mi_haar = mutual_information_matrix(haar)
    upper_haar = mi_haar[np.triu_indices(n, k=1)]
    haar_coherence = float(upper_haar.max() / upper_haar.mean()) if upper_haar.mean() > 1e-10 else 0.0

    print(f"\n  {'Phase':>6} {'SRE':>6} {'MI_max':>7} {'MI_mean':>8} "
          f"{'Coherence':>10} {'Variance':>9}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['phase']:6.3f} {r['sre']:6.3f} {r['mi_max']:7.4f} "
              f"{r['mi_mean']:8.5f} {r['coherence']:10.3f} {r['variance']:9.6f}")
    print(f"  {'Haar':>6} {'max':>6} {float(upper_haar.max()):7.4f} "
          f"{float(upper_haar.mean()):8.5f} {haar_coherence:10.3f} "
          f"{float(upper_haar.var()):9.6f}")

    # Verdict: if coherence stays high as magic increases, geometry is structured (not chaotic)
    mid_coherence = results[len(results)//2]["coherence"]
    print(f"\n  VERDICT: Structured magic coherence = {mid_coherence:.2f}, "
          f"Haar chaos coherence = {haar_coherence:.2f}")
    if mid_coherence > 1.5 * haar_coherence:
        print("  → Structured magic creates COHERENT geometry (not just noise). "
              "Hypothesis SURVIVES.")
    else:
        print("  → Magic-induced geometry is indistinguishable from random. "
              "May be a confound.")

    return results


def falsification_test_4(n: int = 6):
    """Test 4: Basis dependence.

    SRE depends on the Pauli frame. Check whether MI deformation (the
    observable we claim is physical) is independent of local basis choice.

    Strategy: Apply random local Cliffords to the state and verify MI is unchanged.
    This should pass by construction (MI is unitarily invariant) but verifies
    our implementation.
    """
    print(f"\n{'='*60}")
    print(f"FALSIFICATION TEST 4: Basis dependence (n={n})")
    print(f"{'='*60}")
    print("  Question: Is MI deformation invariant under local basis changes?")

    rng = np.random.default_rng(42)

    # Build a magic state
    hg = Hypergraph(n)
    for i in range(n):
        hg.add_edge((i, (i + 1) % n))
    for i in range(0, n - 2, 2):
        hg.add_edge((i, i + 1, i + 2))
    state_orig = HypergraphState(hg).prepare()

    cliffords_1q = [hadamard, s_gate, hadamard @ s_gate,
                    s_gate @ hadamard, hadamard @ s_gate @ hadamard]

    mi_orig = mutual_information_matrix(state_orig)
    sre_orig = stabilizer_renyi_entropy(state_orig)
    br_orig = BackreactionObservable(state_orig).measure()

    print(f"\n  Original state: SRE={sre_orig:.4f}, MI-deform={br_orig.mi_deformation:.4f}")

    # Apply random local Cliffords and check invariance
    n_trials = 5
    max_mi_diff = 0.0
    max_sre_diff = 0.0

    for trial in range(n_trials):
        state_rotated = state_orig
        for q in range(n):
            U = cliffords_1q[rng.integers(len(cliffords_1q))]
            state_rotated = state_rotated.apply_gate(U, [q])

        mi_rotated = mutual_information_matrix(state_rotated)
        sre_rotated = stabilizer_renyi_entropy(state_rotated)
        br_rotated = BackreactionObservable(state_rotated).measure()

        mi_diff = np.linalg.norm(mi_rotated - mi_orig, "fro")
        sre_diff = abs(sre_rotated - sre_orig)
        max_mi_diff = max(max_mi_diff, mi_diff)
        max_sre_diff = max(max_sre_diff, sre_diff)

        print(f"  Trial {trial+1}: SRE={sre_rotated:.4f} (Δ={sre_diff:.2e}), "
              f"MI-deform={br_rotated.mi_deformation:.4f} (ΔMI={mi_diff:.2e})")

    print(f"\n  VERDICT:")
    print(f"    Max MI matrix change: {max_mi_diff:.2e} (should be ~0)")
    print(f"    Max SRE change: {max_sre_diff:.2e} (may vary — SRE is basis-dependent)")
    if max_mi_diff < 1e-10:
        print("  → MI deformation is FRAME-INDEPENDENT (unitarily invariant). "
              "Hypothesis SURVIVES.")
    else:
        print("  → MI deformation varies with basis. Implementation issue.")

    if max_sre_diff > 0.01:
        print(f"  NOTE: SRE varies by up to {max_sre_diff:.4f} under local Cliffords.")
        print("    This is expected — SRE is defined relative to the Pauli frame.")
        print("    The physical observable (MI deformation) is the invariant quantity.")


def falsification_test_5(n: int = 10):
    """Test 5: Locality recovery.

    Check that MI decays with graph distance for magic states.
    Also measures Ollivier-Ricci curvature and spectral dimension.

    Now scales to n=10-12 thanks to FWHT-accelerated SRE.
    """
    print(f"\n{'='*60}")
    print(f"FALSIFICATION TEST 5: Locality recovery (n={n})")
    print(f"{'='*60}")
    print("  Question: Does MI decay with graph distance (locality)?")

    hg = Hypergraph(n)
    for i in range(n):
        hg.add_edge((i, (i + 1) % n))
    for i in range(0, n - 2, 2):
        hg.add_edge((i, i + 1, i + 2))
    state = HypergraphState(hg).prepare()
    mi = mutual_information_matrix(state)

    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            d = min(abs(i - j), n - abs(i - j))
            if d not in distances:
                distances[d] = []
            distances[d].append(mi[i, j])

    print(f"\n  Ring distance → Mean MI:")
    locality_holds = True
    prev_mean = float("inf")
    for d in sorted(distances.keys()):
        mean_mi = np.mean(distances[d])
        std_mi = np.std(distances[d])
        decay_ok = mean_mi <= prev_mean + 0.01
        marker = "✓" if decay_ok else "✗"
        print(f"    d={d}: MI = {mean_mi:.4f} ± {std_mi:.4f} {marker}")
        if not decay_ok:
            locality_holds = False
        prev_mean = mean_mi

    # Geometric observables
    curv = average_curvature(state, threshold=0.01)
    spec_dim = spectral_dimension(state)
    sre = stabilizer_renyi_entropy(state)
    print(f"\n  SRE = {sre:.4f}")
    print(f"  Ollivier-Ricci curvature (mean) = {curv:.4f}")
    print(f"  Spectral dimension = {spec_dim:.4f}")

    print(f"\n  VERDICT:")
    if locality_holds:
        print("  → MI approximately decays with distance. "
              "Emergent geometry is LOCAL. Hypothesis SURVIVES.")
    else:
        print("  → MI does NOT decay with distance. Nonlocal effects detected.")

    return {
        "distances": {str(d): {"mean": float(np.mean(v)), "std": float(np.std(v))}
                      for d, v in distances.items()},
        "curvature": curv, "spectral_dim": spec_dim, "sre": sre,
        "locality_holds": locality_holds,
    }


def falsification_test_6(n: int = 8):
    """Test 6: Universality of coupling + hyperedge-order sweep.

    Tests whether different magic sources and different hyperedge orders
    (k=2,3,4,5) produce comparable geometry per unit magic.

    Key hypergraph-specific test: does the coupling constant K depend
    on the order k of the non-Clifford interaction?
    """
    print(f"\n{'='*60}")
    print(f"FALSIFICATION TEST 6: Universality of coupling (n={n})")
    print(f"{'='*60}")
    print("  Question: Do different magic sources/orders create comparable geometry?")

    results = []

    # --- Source comparison (same as before but at larger n) ---

    hg1 = Hypergraph(n)
    for i in range(n):
        hg1.add_edge((i, (i + 1) % n))
    for i in range(0, n - 2, 2):
        hg1.add_edge((i, i + 1, i + 2))
    state_ccz = HypergraphState(hg1).prepare()

    hg2 = Hypergraph(n)
    for i in range(n):
        hg2.add_edge((i, (i + 1) % n))
    for i in range(0, n - 2, 2):
        hg2.add_edge((i, i + 1, i + 2), np.pi / 3)
    state_cp = HypergraphState(hg2).prepare()

    ring_state = GraphState.ring(n).prepare()
    state_local_t = ring_state
    for q in range(n):
        state_local_t = state_local_t.apply_gate(t_gate, [q])

    state_mixed = GraphState.ring(n).prepare()
    for q in range(0, n, 2):
        state_mixed = state_mixed.apply_gate(t_gate, [q])

    cases = [
        ("CCZ (3-body, phase=π)", state_ccz),
        ("CP(π/3) (3-body, phase=π/3)", state_cp),
        ("Ring + all-T (local magic)", state_local_t),
        ("Ring + alternating-T", state_mixed),
    ]

    for label, state in cases:
        sre = stabilizer_renyi_entropy(state)
        nl = nonlocal_magic(state)
        br = BackreactionObservable(state).measure()
        results.append({
            "label": label, "sre": sre, "nonlocal_magic": nl,
            "mi_deformation": br.mi_deformation, "k": "mixed",
        })
        print(f"\n  {label}:")
        print(f"    SRE = {sre:.4f}, NL-magic = {nl:.4f}")
        print(f"    MI deformation = {br.mi_deformation:.4f}")

    # --- Hyperedge-order sweep (k=2,3,4,5) ---

    print(f"\n  --- Hyperedge-order sweep ---")
    k_results = []
    for k in [2, 3, 4, 5]:
        if k > n:
            continue
        hg = Hypergraph(n)
        for i in range(n):
            hg.add_edge((i, (i + 1) % n))
        for i in range(0, n - k + 1, k):
            edge = tuple(range(i, i + k))
            if k == 2:
                hg.add_edge(edge, np.pi / 3)
            else:
                hg.add_edge(edge)
        state = HypergraphState(hg).prepare()
        sre = stabilizer_renyi_entropy(state)
        nl = nonlocal_magic(state)
        br = BackreactionObservable(state).measure()

        entry = {
            "label": f"k={k}-body hyperedge", "sre": sre,
            "nonlocal_magic": nl, "mi_deformation": br.mi_deformation,
            "k": k,
        }
        k_results.append(entry)
        results.append(entry)

        efficiency = br.mi_deformation / sre if sre > 0.01 else 0
        print(f"  k={k}: SRE={sre:.4f}, NL-magic={nl:.4f}, "
              f"MI-deform={br.mi_deformation:.4f}, efficiency={efficiency:.4f}")

    # Universality check
    efficiencies = [r["mi_deformation"] / r["sre"] for r in results if r["sre"] > 0.01]
    if efficiencies:
        cv = np.std(efficiencies) / np.mean(efficiencies) if np.mean(efficiencies) > 0 else 0
        print(f"\n  Overall coefficient of variation: {cv:.3f}")
        if cv < 0.5:
            print("  → Coupling is approximately UNIVERSAL across sources and orders.")
        else:
            print("  → Coupling is SOURCE/ORDER-DEPENDENT.")

    # Check k-dependence specifically
    k_effs = [(r["k"], r["mi_deformation"] / r["sre"]) for r in k_results if r["sre"] > 0.01]
    if len(k_effs) >= 2:
        ks, effs = zip(*k_effs)
        k_cv = np.std(effs) / np.mean(effs) if np.mean(effs) > 0 else 0
        print(f"\n  Hyperedge-order dependence:")
        for kk, eff in k_effs:
            print(f"    k={kk}: efficiency = {eff:.4f}")
        print(f"    CoV across k: {k_cv:.3f}")

    return results


def falsification_test_7(n_values: list[int] | None = None):
    """Test 7: Scaling with geometric observables.

    Now scales to n=12 with FWHT-accelerated SRE.
    Includes Ollivier-Ricci curvature and spectral dimension at each n.
    """
    if n_values is None:
        n_values = [4, 6, 8, 10, 12]

    print(f"\n{'='*60}")
    print(f"FALSIFICATION TEST 7: Scaling (n={n_values})")
    print(f"{'='*60}")
    print("  Question: Does the magic-geometry effect persist with system size?")

    results = []
    for n in n_values:
        import time
        t0 = time.perf_counter()

        hg = Hypergraph(n)
        for i in range(n):
            hg.add_edge((i, (i + 1) % n))
        for i in range(0, n - 2, 2):
            hg.add_edge((i, i + 1, i + 2))
        state = HypergraphState(hg).prepare()

        sre = stabilizer_renyi_entropy(state)
        br = BackreactionObservable(state).measure()
        curv = average_curvature(state, threshold=0.01)
        spec_dim = spectral_dimension(state)

        sre_per_qubit = sre / n
        mi_deform_per_pair = br.mi_deformation / (n * (n - 1) / 2)
        dt = time.perf_counter() - t0

        results.append({
            "n": n, "sre": sre, "sre_per_qubit": sre_per_qubit,
            "mi_deformation": br.mi_deformation,
            "mi_deform_per_pair": mi_deform_per_pair,
            "curvature": curv, "spectral_dim": spec_dim,
            "time_s": dt,
        })
        print(f"\n  n={n}: SRE={sre:.4f} ({sre_per_qubit:.4f}/qubit), "
              f"MI-deform={br.mi_deformation:.4f} ({mi_deform_per_pair:.4f}/pair)")
        print(f"    Curvature={curv:.4f}, SpectralDim={spec_dim:.4f}, time={dt:.1f}s")

    # Scaling analysis
    print(f"\n  Scaling analysis:")
    mi_per_pair_values = [r["mi_deform_per_pair"] for r in results]
    if mi_per_pair_values[-1] > 0.5 * mi_per_pair_values[0]:
        print("  → MI deformation per pair is STABLE or growing. Hypothesis SURVIVES.")
    else:
        print("  → MI deformation per pair DECAYS with n. May be a finite-size artifact.")

    # Plot with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ns = [r["n"] for r in results]

    axes[0].plot(ns, [r["mi_deformation"] for r in results], "o-", label="MI deformation")
    axes[0].plot(ns, [r["sre"] for r in results], "s--", label="SRE")
    axes[0].set_xlabel("System size n")
    axes[0].set_ylabel("Value")
    axes[0].set_title("Raw observables vs system size")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(ns, [r["mi_deform_per_pair"] for r in results], "o-", label="MI-deform/pair")
    axes[1].plot(ns, [r["sre_per_qubit"] for r in results], "s--", label="SRE/qubit")
    axes[1].set_xlabel("System size n")
    axes[1].set_ylabel("Value per unit")
    axes[1].set_title("Intensive quantities")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(ns, [r["curvature"] for r in results], "^-", color="tab:green", label="Curvature")
    ax2r = axes[2].twinx()
    ax2r.plot(ns, [r["spectral_dim"] for r in results], "v--", color="tab:orange", label="Spectral dim")
    axes[2].set_xlabel("System size n")
    axes[2].set_ylabel("Ollivier-Ricci curvature", color="tab:green")
    ax2r.set_ylabel("Spectral dimension", color="tab:orange")
    axes[2].set_title("Geometric observables vs n")
    lines1, labels1 = axes[2].get_legend_handles_labels()
    lines2, labels2 = ax2r.get_legend_handles_labels()
    axes[2].legend(lines1 + lines2, labels1 + labels2)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "scaling_test.png", dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {OUTPUT_DIR / 'scaling_test.png'}")

    return results


def save_results(all_results: dict, filename: str = "falsification_results.json"):
    """Save all test results to JSON for later analysis."""
    path = OUTPUT_DIR / filename
    serializable = {}
    for key, val in all_results.items():
        if isinstance(val, list):
            serializable[key] = val
        else:
            serializable[key] = str(val)

    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
    print(f"\n  Results saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Falsification tests for magic-gravity hypothesis")
    parser.add_argument("-n", type=int, default=8, help="Default system size")
    parser.add_argument("--n-max", type=int, default=12, help="Max n for scaling test")
    args = parser.parse_args()
    n = args.n

    all_results = {}
    all_results["test2"] = falsification_test_2(n)
    all_results["test3"] = falsification_test_3(n)
    falsification_test_4(n)
    all_results["test5"] = falsification_test_5(max(n, 10))
    all_results["test6"] = falsification_test_6(max(n, 8))

    scaling_ns = [4, 6, 8, 10]
    if args.n_max >= 12:
        scaling_ns.append(12)
    all_results["test7"] = falsification_test_7(scaling_ns)

    save_results(all_results)

    result = ExperimentResult(
        name="falsification_tests",
        params={"n": n, "n_max": args.n_max},
        data=all_results,
    )
    result.save(OUTPUT_DIR)

    print(f"\n\nAll falsification tests complete. Output in {OUTPUT_DIR}")
