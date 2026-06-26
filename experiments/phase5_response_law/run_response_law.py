"""Phase 5: Response Law Numerics.

Tests whether there is a quantitative law relating non-local magic to
geometric deformation (backreaction). Specifically:

  B(θ) = K · M_NL(θ)^α + ...

where K is the response coefficient and α is the scaling exponent.

Measures:
- K_i (response coefficient) via regression of B vs M_NL across state families
- Checks linearity, monotonicity, or absence of relationship
- Reports R² and scaling exponent
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from mqhg.states.hypergraph import HypergraphState, Hypergraph
from mqhg.measures.entanglement import mutual_information_matrix
from mqhg.measures.magic import stabilizer_renyi_entropy, nonlocal_magic
from mqhg.analysis.backreaction import BackreactionObservable


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


def multi_family_response(n: int = 6):
    """Test response law across multiple state families.

    If K is approximately universal, the law is general.
    If K depends strongly on the family, it's structure-specific.
    """
    print(f"\n{'='*60}")
    print(f"MULTI-FAMILY RESPONSE (n={n})")
    print(f"{'='*60}")

    families = {}

    # Family 1: Ring + 3-body (overlapping triplets)
    base_ring = [(i, (i + 1) % n) for i in range(n)]
    magic_overlap = [(i, (i + 1) % n, (i + 2) % n) for i in range(0, n, 2)]

    # Family 2: Ring + 3-body (non-overlapping triplets)
    magic_nonoverlap = [(0, 1, 2), (3, 4, 5)] if n >= 6 else [(0, 1, 2)]

    # Family 3: Complete graph + 3-body
    base_complete = [(i, j) for i in range(n) for j in range(i + 1, n)]
    magic_complete = [(0, 1, 2), (2, 3, 4)]

    family_configs = [
        ("Ring + overlapping CCZ", base_ring, magic_overlap),
        ("Ring + non-overlapping CCZ", base_ring, magic_nonoverlap),
        ("Complete + CCZ", base_complete, magic_complete),
    ]

    all_family_data = []

    for name, base, magic_e in family_configs:
        phases = np.linspace(0.1, np.pi, 15)
        fam_results = []

        for phi in phases:
            hg = Hypergraph(n)
            for e in base:
                hg.add_edge(e, np.pi)
            for e in magic_e:
                hg.add_edge(e, phi)
            state = HypergraphState(hg).prepare()

            sre = stabilizer_renyi_entropy(state)
            nl = nonlocal_magic(state)
            br = BackreactionObservable(state).measure()
            fam_results.append({"magic": nl, "deformation": br.mi_deformation})

        # Linear fit for this family
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

    # Plot all families
    fig, ax = plt.subplots(figsize=(10, 7))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_family_data)))

    for idx, (name, m, d, coeffs) in enumerate(all_family_data):
        ax.scatter(m, d, color=colors[idx], label=name, alpha=0.7)
        x_fit = np.linspace(m.min(), m.max(), 50)
        ax.plot(x_fit, np.polyval(coeffs, x_fit), "--", color=colors[idx], alpha=0.5)

    ax.set_xlabel("Non-local Magic (SRE₂)")
    ax.set_ylabel("MI Deformation (Backreaction)")
    ax.set_title(f"Response Law Across Families (n={n})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(OUTPUT_DIR / "response_law_families.png", dpi=150, bbox_inches="tight")
    print(f"\n  Plot saved: {OUTPUT_DIR / 'response_law_families.png'}")

    # Universality check
    k_values = [f["K"] for f in families.values()]
    if len(k_values) >= 2:
        cv = np.std(k_values) / np.mean(k_values) if np.mean(k_values) > 0 else 0
        print(f"\n  Response coefficients: {[f'{k:.4f}' for k in k_values]}")
        print(f"  Coefficient of variation: {cv:.3f}")
        if cv < 0.3:
            print("  → Response coefficient is approximately UNIVERSAL across families.")
        else:
            print("  → Response coefficient is FAMILY-DEPENDENT (structure matters).")

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


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 6

    # Main sweep
    results = sweep_response_law(n, n_phases=30)
    analysis = analyze_response_law(results)
    plot_response_law(results, analysis)

    # Multi-family comparison
    families = multi_family_response(n)

    # Save
    output = {"sweep": results, "analysis": analysis}
    with open(OUTPUT_DIR / "response_law_data.json", "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Data saved to {OUTPUT_DIR / 'response_law_data.json'}")

    print(f"\n\nPhase 5 response law analysis complete. Output in {OUTPUT_DIR}")
