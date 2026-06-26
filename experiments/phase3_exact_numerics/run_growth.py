"""Growth model phase diagram experiment.

Sweeps p_magic (non-Clifford gate rate) to map the predicted
rigid / deformable / chaotic phase transition in emergent geometry.

At each p_magic, runs multiple random circuit seeds and measures:
- SRE (total magic)
- Non-local magic
- Geometry smoothness (inverse CoV of MI distances)
- Mean entanglement entropy
- Ollivier-Ricci curvature (emergent spacetime curvature)
- Spectral dimension

The key prediction: there exists a "Goldilocks" regime of moderate
p_magic where geometry is both non-trivial and smooth.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mqhg.models.growth import RandomCircuitGrowth, GrowthConfig
from mqhg.core.statevector import Statevector
from mqhg.core.gates import cz, ccz, t_gate, hadamard, s_gate
from mqhg.measures.entanglement import mutual_information_matrix, subsystem_entropy
from mqhg.measures.magic import stabilizer_renyi_entropy, nonlocal_magic
from mqhg.measures.geometry import (
    mutual_info_distance_matrix,
    average_curvature,
    spectral_dimension,
)
from mqhg.io.results import ExperimentResult

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

_CLIFFORD_1Q = [hadamard, s_gate, hadamard @ s_gate, s_gate @ hadamard]


def _run_single_circuit(n: int, depth: int, p_magic: float, seed: int) -> dict:
    """Run one random circuit and return final-state observables."""
    rng = np.random.default_rng(seed)
    state = Statevector.plus_n(n)

    for _step in range(depth):
        for q in range(n):
            if rng.random() < p_magic:
                state = state.apply_gate(t_gate, [q])
            else:
                gate = _CLIFFORD_1Q[rng.integers(len(_CLIFFORD_1Q))]
                state = state.apply_gate(gate, [q])
        qubits = list(range(n))
        rng.shuffle(qubits)
        for i in range(0, n - 1, 2):
            state = state.apply_gate(cz, [qubits[i], qubits[i + 1]])
        if n >= 3 and rng.random() < p_magic:
            targets = sorted(rng.choice(n, size=3, replace=False).tolist())
            state = state.apply_gate(ccz, targets)

    sre = stabilizer_renyi_entropy(state)
    nl = nonlocal_magic(state)
    entropies = [subsystem_entropy(state, [q]) for q in range(n)]
    mean_ent = float(np.mean(entropies))

    dist = mutual_info_distance_matrix(state)
    upper = dist[np.triu_indices(n, k=1)]
    smoothness = float(np.mean(upper) / np.std(upper)) if np.std(upper) > 1e-10 else float("inf")

    curv = average_curvature(state, threshold=0.01)
    spec_dim = spectral_dimension(state)

    return {
        "sre": sre,
        "nonlocal_magic": nl,
        "mean_entropy": mean_ent,
        "smoothness": smoothness,
        "curvature": curv,
        "spectral_dim": spec_dim,
    }


def sweep_phase_diagram(
    n: int = 8,
    depth: int = 10,
    n_seeds: int = 5,
    p_values: list[float] | None = None,
) -> list[dict]:
    """Sweep p_magic and collect averaged observables."""
    if p_values is None:
        p_values = [
            0.0, 0.01, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2,
            0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
        ]

    results = []
    total = len(p_values) * n_seeds
    done = 0

    for p in p_values:
        seed_results = []
        for seed in range(n_seeds):
            obs = _run_single_circuit(n, depth, p, seed)
            seed_results.append(obs)
            done += 1
            print(f"  [{done}/{total}] p_magic={p:.2f} seed={seed}: "
                  f"SRE={obs['sre']:.3f}, smooth={obs['smoothness']:.2f}, "
                  f"curv={obs['curvature']:.3f}")

        keys = seed_results[0].keys()
        avg = {}
        for key in keys:
            vals = [r[key] for r in seed_results]
            avg[f"{key}_mean"] = float(np.mean(vals))
            avg[f"{key}_std"] = float(np.std(vals))
        avg["p_magic"] = p
        results.append(avg)

    return results


def classify_phases(results: list[dict]) -> dict:
    """Classify the phase diagram into rigid / deformable / chaotic regimes."""
    p_vals = [r["p_magic"] for r in results]
    smoothness = [r["smoothness_mean"] for r in results]
    sre = [r["sre_mean"] for r in results]
    curvature = [r["curvature_mean"] for r in results]

    smooth_arr = np.array(smoothness)
    sre_arr = np.array(sre)

    # Rigid: low SRE, high or infinite smoothness
    # Deformable: moderate SRE, moderate-high smoothness, non-zero curvature
    # Chaotic: high SRE, low smoothness

    sre_threshold_low = 0.1
    sre_threshold_high = np.percentile(sre_arr[sre_arr > 0], 75) if np.any(sre_arr > 0) else 1.0
    smooth_threshold = np.median(smooth_arr[np.isfinite(smooth_arr)]) if np.any(np.isfinite(smooth_arr)) else 1.0

    phases = {}
    rigid_end = 0.0
    chaotic_start = 1.0
    for r in results:
        p = r["p_magic"]
        if r["sre_mean"] < sre_threshold_low:
            phases[p] = "rigid"
            rigid_end = max(rigid_end, p)
        elif r["smoothness_mean"] < smooth_threshold * 0.5:
            phases[p] = "chaotic"
            chaotic_start = min(chaotic_start, p)
        else:
            phases[p] = "deformable"

    return {
        "phases": phases,
        "rigid_boundary": rigid_end,
        "chaotic_boundary": chaotic_start,
        "sre_threshold_low": sre_threshold_low,
        "sre_threshold_high": sre_threshold_high,
        "smooth_threshold": smooth_threshold,
    }


def plot_phase_diagram(results: list[dict], classification: dict, n: int):
    """Generate phase diagram plot."""
    p_vals = [r["p_magic"] for r in results]
    sre_mean = [r["sre_mean"] for r in results]
    sre_std = [r["sre_std"] for r in results]
    smooth_mean = [r["smoothness_mean"] for r in results]
    smooth_std = [r["smoothness_std"] for r in results]
    curv_mean = [r["curvature_mean"] for r in results]
    curv_std = [r["curvature_std"] for r in results]
    nl_mean = [r["nonlocal_magic_mean"] for r in results]
    spec_dim_mean = [r["spectral_dim_mean"] for r in results]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"Growth Model Phase Diagram (n={n})", fontsize=14, fontweight="bold")

    phase_colors = {"rigid": "#3498db", "deformable": "#2ecc71", "chaotic": "#e74c3c"}
    bg_colors = [phase_colors.get(classification["phases"].get(p, "deformable"), "#ccc") for p in p_vals]

    # Panel 1: SRE vs p_magic
    axes[0, 0].errorbar(p_vals, sre_mean, yerr=sre_std, fmt="o-", color="tab:red", capsize=3)
    axes[0, 0].set_ylabel("SRE (total magic)")
    axes[0, 0].set_title("Magic content")
    axes[0, 0].grid(True, alpha=0.3)

    # Panel 2: Geometry smoothness vs p_magic
    finite_smooth = [(p, s, e) for p, s, e in zip(p_vals, smooth_mean, smooth_std) if np.isfinite(s)]
    if finite_smooth:
        ps, ss, es = zip(*finite_smooth)
        axes[0, 1].errorbar(ps, ss, yerr=es, fmt="s-", color="tab:blue", capsize=3)
    axes[0, 1].set_ylabel("Geometry smoothness")
    axes[0, 1].set_title("Smoothness (mean/std of MI distances)")
    axes[0, 1].grid(True, alpha=0.3)

    # Panel 3: Curvature vs p_magic
    axes[0, 2].errorbar(p_vals, curv_mean, yerr=curv_std, fmt="^-", color="tab:green", capsize=3)
    axes[0, 2].set_ylabel("Ollivier-Ricci curvature")
    axes[0, 2].set_title("Emergent curvature")
    axes[0, 2].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0, 2].grid(True, alpha=0.3)

    # Panel 4: Non-local magic
    axes[1, 0].plot(p_vals, nl_mean, "D-", color="tab:purple")
    axes[1, 0].set_ylabel("Non-local magic")
    axes[1, 0].set_xlabel("p_magic")
    axes[1, 0].set_title("Non-local magic")
    axes[1, 0].grid(True, alpha=0.3)

    # Panel 5: Spectral dimension
    axes[1, 1].plot(p_vals, spec_dim_mean, "v-", color="tab:orange")
    axes[1, 1].set_ylabel("Spectral dimension")
    axes[1, 1].set_xlabel("p_magic")
    axes[1, 1].set_title("Spectral dimension")
    axes[1, 1].grid(True, alpha=0.3)

    # Panel 6: Phase classification
    phase_map = {"rigid": 0, "deformable": 1, "chaotic": 2}
    phase_vals = [phase_map.get(classification["phases"].get(p, "deformable"), 1) for p in p_vals]
    axes[1, 2].scatter(p_vals, phase_vals, c=bg_colors, s=100, edgecolors="k", zorder=5)
    axes[1, 2].set_yticks([0, 1, 2])
    axes[1, 2].set_yticklabels(["Rigid", "Deformable", "Chaotic"])
    axes[1, 2].set_xlabel("p_magic")
    axes[1, 2].set_title("Phase classification")
    rb = classification["rigid_boundary"]
    cb = classification["chaotic_boundary"]
    axes[1, 2].axvline(x=rb, color="blue", linestyle="--", alpha=0.5, label=f"Rigid boundary ~{rb:.2f}")
    axes[1, 2].axvline(x=cb, color="red", linestyle="--", alpha=0.5, label=f"Chaotic boundary ~{cb:.2f}")
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "growth_phase_diagram.png", dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {OUTPUT_DIR / 'growth_phase_diagram.png'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Growth model phase diagram")
    parser.add_argument("-n", "--n-qubits", type=int, default=8)
    parser.add_argument("-d", "--depth", type=int, default=10)
    parser.add_argument("-s", "--seeds", type=int, default=5)
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"GROWTH MODEL PHASE DIAGRAM (n={args.n_qubits}, depth={args.depth})")
    print(f"{'='*60}")

    results = sweep_phase_diagram(
        n=args.n_qubits,
        depth=args.depth,
        n_seeds=args.seeds,
    )

    classification = classify_phases(results)
    print(f"\n  Phase boundaries:")
    print(f"    Rigid → Deformable: p_magic ≈ {classification['rigid_boundary']:.2f}")
    print(f"    Deformable → Chaotic: p_magic ≈ {classification['chaotic_boundary']:.2f}")

    plot_phase_diagram(results, classification, args.n_qubits)

    result = ExperimentResult(
        name="growth_phase_diagram",
        params={"n_qubits": args.n_qubits, "depth": args.depth, "n_seeds": args.seeds},
        data={"sweep": results, "classification": classification},
    )
    out_dir = result.save(OUTPUT_DIR)
    print(f"  Data saved to {out_dir}")

    with open(OUTPUT_DIR / "growth_phase_diagram.json", "w") as f:
        json.dump({"sweep": results, "classification": classification}, f, indent=2, default=str)

    print(f"\n  Growth model phase diagram complete.")
