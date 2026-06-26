"""Phase 5: Response Law Numerics (Scaled).

Tests the quantitative law B(θ) = K · M_NL(θ)^α across:
- Multiple system sizes (n=6,8,10,12)
- Multiple hypergraph families (varying k, topologies)
- High phase resolution (50 points)

Key questions:
1. Is K approximately universal across families?
2. How does K scale with system size n?
3. Does the scaling exponent α depend on hyperedge order k?
"""

import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mqhg.states.hypergraph import HypergraphState, Hypergraph
from mqhg.states.graph import GraphState
from mqhg.measures.entanglement import mutual_information_matrix
from mqhg.measures.magic import stabilizer_renyi_entropy, nonlocal_magic
from mqhg.measures.geometry import average_curvature, spectral_dimension
from mqhg.analysis.backreaction import BackreactionObservable
from mqhg.io.results import ExperimentResult


OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def sweep_response_law(n: int = 6, n_phases: int = 30):
    """Fine-grained sweep of magic phase to extract the response law.

    Uses ring + 3-body hypergraph with varying phase θ.
    At each θ: compute SRE, non-local magic, and MI deformation.
    """
    print(f"\n{'='*60}")
    print(f"RESPONSE LAW: Fine sweep (n={n}, {n_phases} points)")
    print(f"{'='*60}")

    base_edges = [(i, (i + 1) % n) for i in range(n)]
    magic_edges = [(i, (i + 1) % n, (i + 2) % n) for i in range(0, n, 2)]

    phases = np.linspace(0, np.pi, n_phases)
    results = []

    for phi in phases:
        hg = Hypergraph(n)
        for e in base_edges:
            hg.add_edge(e, np.pi)
        for e in magic_edges:
            hg.add_edge(e, phi)
        state = HypergraphState(hg).prepare()

        sre = stabilizer_renyi_entropy(state)
        nl = nonlocal_magic(state)
        br = BackreactionObservable(state).measure()

        results.append({
            "phase": float(phi),
            "sre": sre,
            "nonlocal_magic": nl,
            "mi_deformation": br.mi_deformation,
            "mi_max": br.mi_max,
            "mi_mean": br.mi_mean,
        })

    return results


def analyze_response_law(results: list[dict]):
    """Fit the response law B = K · M^α and report statistics."""
    print(f"\n{'='*60}")
    print("RESPONSE LAW ANALYSIS")
    print(f"{'='*60}")

    magic = np.array([r["nonlocal_magic"] for r in results])
    deformation = np.array([r["mi_deformation"] for r in results])

    # Filter out zero-magic points for log fitting
    mask = (magic > 0.01) & (deformation > 0.001)
    m_pos = magic[mask]
    d_pos = deformation[mask]

    if len(m_pos) < 3:
        print("  Insufficient nonzero data points for fitting.")
        return {}

    # Linear fit: B = a * M + b
    coeffs_lin = np.polyfit(m_pos, d_pos, 1)
    d_pred_lin = np.polyval(coeffs_lin, m_pos)
    ss_res_lin = np.sum((d_pos - d_pred_lin) ** 2)
    ss_tot = np.sum((d_pos - d_pos.mean()) ** 2)
    r2_linear = 1 - ss_res_lin / ss_tot if ss_tot > 1e-10 else 0

    print(f"\n  Linear fit: B = {coeffs_lin[0]:.4f} · M + {coeffs_lin[1]:.4f}")
    print(f"    R² = {r2_linear:.4f}")
    print(f"    Response coefficient K = {coeffs_lin[0]:.4f}")

    # Power-law fit: log(B) = α·log(M) + log(K)
    log_m = np.log(m_pos)
    log_d = np.log(d_pos)
    coeffs_pow = np.polyfit(log_m, log_d, 1)
    alpha = coeffs_pow[0]
    K_pow = np.exp(coeffs_pow[1])
    d_pred_pow = K_pow * m_pos ** alpha
    ss_res_pow = np.sum((d_pos - d_pred_pow) ** 2)
    r2_power = 1 - ss_res_pow / ss_tot if ss_tot > 1e-10 else 0

    print(f"\n  Power-law fit: B = {K_pow:.4f} · M^{alpha:.4f}")
    print(f"    R² = {r2_power:.4f}")
    print(f"    Scaling exponent α = {alpha:.4f}")

    # Monotonicity check
    diffs = np.diff(d_pos)
    n_monotone = np.sum(diffs >= -1e-10)
    monotone_frac = n_monotone / len(diffs) if len(diffs) > 0 else 0

    print(f"\n  Monotonicity: {n_monotone}/{len(diffs)} segments increasing "
          f"({monotone_frac*100:.0f}%)")

    # Verdict
    print(f"\n  CONCLUSION:")
    if r2_linear > 0.9:
        print(f"    Strong LINEAR relationship (R²={r2_linear:.3f}). "
              f"Response coefficient K = {coeffs_lin[0]:.4f}")
    elif r2_power > 0.9:
        print(f"    Strong POWER-LAW relationship (R²={r2_power:.3f}, α={alpha:.2f}). "
              f"Not linear but monotone.")
    elif monotone_frac > 0.8:
        print(f"    MONOTONE but not well-fit by simple law. "
              f"Complex relationship.")
    else:
        print(f"    NO clear relationship between magic and geometry.")

    return {
        "K_linear": float(coeffs_lin[0]),
        "intercept": float(coeffs_lin[1]),
        "r2_linear": float(r2_linear),
        "alpha": float(alpha),
        "K_power": float(K_pow),
        "r2_power": float(r2_power),
        "monotone_fraction": float(monotone_frac),
    }


def multi_family_response(n: int = 8, n_phases: int = 30):
    """Test response law across expanded hypergraph families.

    Families include different hyperedge orders (k=3,4,5),
    different topologies (ring, star, random), and varying overlap.
    """
    print(f"\n{'='*60}")
    print(f"MULTI-FAMILY RESPONSE (n={n}, {n_phases} phases)")
    print(f"{'='*60}")

    families = {}
    base_ring = [(i, (i + 1) % n) for i in range(n)]

    family_configs = [
        ("Ring + k=3 overlapping",
         base_ring,
         [(i, (i + 1) % n, (i + 2) % n) for i in range(0, n, 2)]),
        ("Ring + k=3 non-overlapping",
         base_ring,
         [(i, i + 1, i + 2) for i in range(0, n - 2, 3)]),
        ("Ring + k=4",
         base_ring,
         [(i, i + 1, i + 2, i + 3) for i in range(0, n - 3, 4)]),
    ]

    if n >= 10:
        family_configs.append((
            "Ring + k=5",
            base_ring,
            [(i, i + 1, i + 2, i + 3, i + 4) for i in range(0, n - 4, 5)],
        ))

    # Random hypergraph family
    rng = np.random.default_rng(42)
    random_edges = []
    for _ in range(n // 3):
        k = rng.integers(3, min(6, n + 1))
        edge = tuple(sorted(rng.choice(n, size=k, replace=False)))
        random_edges.append(edge)
    family_configs.append(("Ring + random k-body", base_ring, random_edges))

    # Star base
    base_star = [(0, i) for i in range(1, n)]
    family_configs.append((
        "Star + k=3",
        base_star,
        [(0, i, i + 1) for i in range(1, n - 1, 2)],
    ))

    all_family_data = []

    for name, base, magic_e in family_configs:
        phases = np.linspace(0.1, np.pi, n_phases)
        fam_results = []

        for phi in phases:
            hg = Hypergraph(n)
            for e in base:
                hg.add_edge(e, np.pi)
            for e in magic_e:
                hg.add_edge(e, phi)
            state = HypergraphState(hg).prepare()

            nl = nonlocal_magic(state)
            br = BackreactionObservable(state).measure()
            fam_results.append({"magic": nl, "deformation": br.mi_deformation, "phase": float(phi)})

        m_vals = np.array([r["magic"] for r in fam_results])
        d_vals = np.array([r["deformation"] for r in fam_results])
        mask = m_vals > 0.01
        if mask.sum() >= 3:
            coeffs = np.polyfit(m_vals[mask], d_vals[mask], 1)
            families[name] = {"K": float(coeffs[0]), "data": fam_results}
            all_family_data.append((name, m_vals[mask], d_vals[mask], coeffs))
            print(f"\n  {name}: K = {coeffs[0]:.4f}")
        else:
            print(f"\n  {name}: insufficient data")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_family_data)))

    for idx, (name, m, d, coeffs) in enumerate(all_family_data):
        ax.scatter(m, d, color=colors[idx], label=name, alpha=0.7, s=30)
        x_fit = np.linspace(m.min(), m.max(), 50)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color=colors[idx], alpha=0.5)

    ax.set_xlabel("Non-local Magic (SRE₂)")
    ax.set_ylabel("MI Deformation (Backreaction)")
    ax.set_title(f"Response Law Across Hypergraph Families (n={n})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "response_law_families.png", dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {OUTPUT_DIR / 'response_law_families.png'}")

    k_values = [f["K"] for f in families.values()]
    if len(k_values) >= 2:
        cv = np.std(k_values) / np.mean(k_values) if np.mean(k_values) > 0 else 0
        print(f"\n  Response coefficients: {[f'{k:.4f}' for k in k_values]}")
        print(f"  Coefficient of variation: {cv:.3f}")
        if cv < 0.3:
            print("  → Response coefficient is approximately UNIVERSAL.")
        else:
            print("  → Response coefficient is FAMILY-DEPENDENT.")

    return families


def plot_response_law(results: list[dict], analysis: dict):
    """Generate publication-quality response law plot."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    phases = [r["phase"] for r in results]
    magic = [r["nonlocal_magic"] for r in results]
    deformation = [r["mi_deformation"] for r in results]
    sre = [r["sre"] for r in results]

    # Panel 1: Observables vs phase
    axes[0].plot(phases, sre, "s-", label="SRE", color="tab:red")
    axes[0].plot(phases, deformation, "o-", label="MI deformation", color="tab:blue")
    axes[0].set_xlabel("Magic phase θ")
    axes[0].set_ylabel("Observable value")
    axes[0].set_title("Observables vs magic injection")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Panel 2: B vs M (the response law)
    axes[1].scatter(magic, deformation, c=phases, cmap="viridis", edgecolors="k", linewidths=0.5)
    if analysis.get("K_linear"):
        m_fit = np.linspace(0, max(magic), 50)
        axes[1].plot(m_fit, analysis["K_linear"] * m_fit + analysis["intercept"],
                     "r--", label=f"Linear: K={analysis['K_linear']:.3f}, R²={analysis['r2_linear']:.3f}")
    axes[1].set_xlabel("Non-local Magic")
    axes[1].set_ylabel("MI Deformation (Backreaction)")
    axes[1].set_title("Response Law: B vs M")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Log-log (power law check)
    mask = (np.array(magic) > 0.01) & (np.array(deformation) > 0.001)
    m_pos = np.array(magic)[mask]
    d_pos = np.array(deformation)[mask]
    if len(m_pos) > 2:
        axes[2].scatter(np.log(m_pos), np.log(d_pos), c="tab:green", edgecolors="k", linewidths=0.5)
        if analysis.get("alpha"):
            log_fit = np.linspace(np.log(m_pos.min()), np.log(m_pos.max()), 50)
            axes[2].plot(log_fit, analysis["alpha"] * log_fit + np.log(analysis["K_power"]),
                         "r--", label=f"α={analysis['alpha']:.2f}, R²={analysis['r2_power']:.3f}")
        axes[2].set_xlabel("log(Non-local Magic)")
        axes[2].set_ylabel("log(MI Deformation)")
        axes[2].set_title("Power-law check (log-log)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "response_law.png", dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {OUTPUT_DIR / 'response_law.png'}")


def multi_n_response(n_values: list[int] | None = None, n_phases: int = 30):
    """Sweep response law across multiple system sizes to test K(n) scaling."""
    if n_values is None:
        n_values = [6, 8, 10]

    print(f"\n{'='*60}")
    print(f"MULTI-N RESPONSE LAW (n={n_values}, {n_phases} phases)")
    print(f"{'='*60}")

    n_results = {}
    for n in n_values:
        print(f"\n  --- n={n} ---")
        results = sweep_response_law(n, n_phases=n_phases)
        analysis = analyze_response_law(results)
        n_results[n] = {"sweep": results, "analysis": analysis}

    # Plot K vs n
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = []
    ks_lin = []
    ks_pow = []
    for n_val, data in n_results.items():
        a = data["analysis"]
        if a.get("K_linear"):
            ns.append(n_val)
            ks_lin.append(a["K_linear"])
            ks_pow.append(a.get("K_power", 0))

    if ns:
        ax.plot(ns, ks_lin, "o-", label="K (linear fit)")
        ax.set_xlabel("System size n")
        ax.set_ylabel("Response coefficient K")
        ax.set_title("Response Coefficient vs System Size")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        fig.savefig(OUTPUT_DIR / "response_K_vs_n.png", dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved: {OUTPUT_DIR / 'response_K_vs_n.png'}")
    plt.close(fig)

    return n_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Response law numerics (scaled)")
    parser.add_argument("-n", type=int, default=8, help="Primary system size")
    parser.add_argument("--n-phases", type=int, default=50, help="Phase resolution")
    parser.add_argument("--multi-n", nargs="+", type=int, default=None,
                        help="System sizes for multi-n sweep")
    args = parser.parse_args()

    # Main sweep at primary n
    results = sweep_response_law(args.n, n_phases=args.n_phases)
    analysis = analyze_response_law(results)
    plot_response_law(results, analysis)

    # Multi-family comparison
    families = multi_family_response(args.n, n_phases=min(args.n_phases, 30))

    # Multi-n sweep
    multi_n_sizes = args.multi_n or [6, 8, 10]
    n_results = multi_n_response(multi_n_sizes, n_phases=min(args.n_phases, 30))

    # Save all
    output = {
        "primary_sweep": results,
        "analysis": analysis,
        "families": {k: v.get("K") for k, v in families.items()},
        "multi_n": {str(k): v["analysis"] for k, v in n_results.items()},
    }
    with open(OUTPUT_DIR / "response_law_data.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    result = ExperimentResult(
        name="response_law",
        params={"n": args.n, "n_phases": args.n_phases, "multi_n": multi_n_sizes},
        data=output,
    )
    result.save(OUTPUT_DIR)

    print(f"\n\nPhase 5 response law analysis complete. Output in {OUTPUT_DIR}")
