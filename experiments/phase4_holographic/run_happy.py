"""Phase 4: Holographic toy-code numerics.

Tests the HaPPY code and approximate holographic code models:
1. Compare perfect stabilizer code vs non-stabilizer code
2. Measure min-cut, boundary entropy, reconstruction fidelity
3. Test whether magic-bearing perturbations produce controlled breakdown
   of perfect code rigidity, resembling backreaction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")

from mqhg.models.happy import HaPPYCode, HaPPYConfig, TensorType
from mqhg.models.approximate import ApproximateHolographicCode, ApproximateCodeConfig
from mqhg.measures.entanglement import subsystem_entropy, mutual_information_matrix
from mqhg.measures.magic import stabilizer_renyi_entropy, nonlocal_magic
from mqhg.analysis.plotting import plot_sweep_results


OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def experiment_happy_comparison():
    """Compare stabilizer vs non-stabilizer HaPPY code."""
    print(f"\n{'='*60}")
    print("HaPPY CODE: Stabilizer vs Non-Stabilizer")
    print(f"{'='*60}")

    code = HaPPYCode()
    logical_0 = np.array([1, 0], dtype=np.complex128)
    logical_plus = np.array([1, 1], dtype=np.complex128) / np.sqrt(2)

    for logical_label, logical_state in [("| 0>", logical_0), ("|+>", logical_plus)]:
        print(f"\n  Logical state: {logical_label}")
        results = code.compare_stabilizer_vs_magic(logical_state)

        for tt, data in results.items():
            print(f"\n    {tt}:")
            print(f"      SRE = {data['sre']:.6f}")
            print(f"      Non-local magic = {data['nonlocal_magic']:.6f}")
            print(f"      Min-cut = {data['min_cut']:.6f}")


def experiment_happy_perturbation():
    """Perturb the HaPPY code and measure geometry changes."""
    print(f"\n{'='*60}")
    print("HaPPY CODE: Perturbation Study")
    print(f"{'='*60}")

    logical_state = np.array([1, 0], dtype=np.complex128)

    for tt in TensorType:
        config = HaPPYConfig(tensor_type=tt)
        code = HaPPYCode(config)
        state = code.encode(logical_state)

        # Boundary entropies
        n_boundary = config.n_boundary
        entropies = {}
        for size in range(1, n_boundary // 2 + 1):
            entropies[size] = code.boundary_entropy(logical_state, list(range(size)))

        print(f"\n  Tensor type: {tt.value}")
        print(f"    Boundary entropies by region size: {entropies}")
        print(f"    Decode fidelity: {code.decode_fidelity(state, logical_state):.6f}")


def experiment_approximate_sweep():
    """Sweep magic_strength in approximate holographic code."""
    print(f"\n{'='*60}")
    print("APPROXIMATE CODE: Magic Strength Sweep")
    print(f"{'='*60}")

    config = ApproximateCodeConfig(n_physical=6, n_logical=1)
    code = ApproximateHolographicCode(config)
    results = code.sweep_magic_strength(n_points=15)

    for r in results:
        print(f"  ε={r['magic_strength']:.3f}: SRE={r['sre']:.4f}, "
              f"NL-magic={r['nonlocal_magic']:.4f}, S_half={r['boundary_entropy_half']:.4f}")

    plot_sweep_results(
        results,
        x_key="magic_strength",
        y_keys=["sre", "nonlocal_magic", "boundary_entropy_half"],
        title="Approximate Holographic Code: Magic Strength Sweep",
        save_path=OUTPUT_DIR / "approximate_code_sweep.png",
    )
    print(f"\n  Plot saved: {OUTPUT_DIR / 'approximate_code_sweep.png'}")


def experiment_reconstruction_error():
    """Measure reconstruction error as function of magic strength."""
    print(f"\n{'='*60}")
    print("APPROXIMATE CODE: Reconstruction Error vs Magic")
    print(f"{'='*60}")

    logical_state = np.array([1, 0], dtype=np.complex128)
    n_physical = 6

    results = []
    for eps in np.linspace(0, 2.0, 12):
        config = ApproximateCodeConfig(n_physical=n_physical, magic_strength=eps)
        code = ApproximateHolographicCode(config)

        # Erase 1 qubit
        err_1 = code.reconstruction_error(logical_state, [0])
        # Erase 2 qubits
        err_2 = code.reconstruction_error(logical_state, [0, 1])

        results.append({
            "magic_strength": eps,
            "recon_error_1qubit": err_1,
            "recon_error_2qubit": err_2,
        })

        print(f"  ε={eps:.3f}: err(1 erased)={err_1:.4f}, err(2 erased)={err_2:.4f}")

    plot_sweep_results(
        results,
        x_key="magic_strength",
        y_keys=["recon_error_1qubit", "recon_error_2qubit"],
        title="Reconstruction Error vs Magic Strength",
        save_path=OUTPUT_DIR / "reconstruction_error.png",
    )
    print(f"\n  Plot saved: {OUTPUT_DIR / 'reconstruction_error.png'}")


if __name__ == "__main__":
    experiment_happy_comparison()
    experiment_happy_perturbation()
    experiment_approximate_sweep()
    experiment_reconstruction_error()

    print(f"\n\nAll Phase 4 experiments complete. Output in {OUTPUT_DIR}")
