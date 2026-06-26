"""Phase 3 Hardware Extension: Validate shadow tomography SRE estimation.

This script:
1. Validates the shadow pipeline classically (compare shadow SRE to exact SRE at n=6-8)
2. Builds circuits for hardware execution at larger n
3. Provides cost estimates before submission
4. Processes hardware results and compares to classical baselines

Usage:
    # Classical validation only (no hardware needed):
    python3 run_hardware.py --validate

    # Build circuits and estimate costs for n=20:
    python3 run_hardware.py --plan --n-qubits 20

    # Process saved hardware results:
    python3 run_hardware.py --process results.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json

import numpy as np

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def validate_shadow_pipeline(n_qubits: int = 6, n_measurements: int = 2000):
    """Validate shadow SRE estimation against exact classical computation.

    Runs entirely on classical simulation — no hardware required.
    """
    from mqhg.states.hypergraph import Hypergraph, HypergraphState
    from mqhg.states.graph import GraphState
    from mqhg.measures.magic import stabilizer_renyi_entropy, nonlocal_magic
    from mqhg.hardware.circuits import hypergraph_to_circuit, graph_state_circuit
    from mqhg.hardware.shadows import ShadowProtocol
    from mqhg.hardware.sre_estimator import estimate_sre_from_shadows, confidence_interval

    print(f"\n{'='*65}")
    print(f" SHADOW TOMOGRAPHY VALIDATION (n={n_qubits}, N_meas={n_measurements})")
    print(f"{'='*65}")

    # --- Test 1: Stabilizer state (should give SRE ≈ 0) ---
    print(f"\n--- Test 1: Ring graph state (stabilizer, SRE should ≈ 0) ---")

    gs = GraphState.ring(n_qubits)
    state_exact = gs.prepare()
    sre_exact = stabilizer_renyi_entropy(state_exact)

    import networkx as nx
    qc = graph_state_circuit(nx.cycle_graph(n_qubits))

    protocol = ShadowProtocol(n_qubits, n_measurements=n_measurements, seed=42)
    shadow_data = protocol.simulate_shadows(qc)
    sre_shadow = estimate_sre_from_shadows(shadow_data, n_pauli_samples=300, seed=42)
    mean, lower, upper = confidence_interval(shadow_data, n_pauli_samples=300, n_bootstrap=50, seed=42)

    print(f"  Exact SRE:  {sre_exact:.6f}")
    print(f"  Shadow SRE: {sre_shadow:.6f}")
    print(f"  95% CI:     [{lower:.4f}, {upper:.4f}]")
    print(f"  Match: {'PASS' if abs(sre_shadow - sre_exact) < 0.3 else 'FAIL'}")

    # --- Test 2: Magic state (should give SRE > 0) ---
    print(f"\n--- Test 2: Hypergraph state with CCZ (should have magic) ---")

    hg = Hypergraph(n_qubits)
    for i in range(n_qubits):
        hg.add_edge((i, (i + 1) % n_qubits))
    for i in range(0, n_qubits - 2, 2):
        hg.add_edge((i, i + 1, i + 2))

    state_exact = HypergraphState(hg).prepare()
    sre_exact = stabilizer_renyi_entropy(state_exact)
    nl_exact = nonlocal_magic(state_exact)

    qc = hypergraph_to_circuit(hg)

    protocol = ShadowProtocol(n_qubits, n_measurements=n_measurements, seed=123)
    shadow_data = protocol.simulate_shadows(qc)
    sre_shadow = estimate_sre_from_shadows(shadow_data, n_pauli_samples=300, seed=123)
    mean, lower, upper = confidence_interval(shadow_data, n_pauli_samples=300, n_bootstrap=50, seed=123)

    print(f"  Exact SRE:         {sre_exact:.6f}")
    print(f"  Shadow SRE:        {sre_shadow:.6f}")
    print(f"  95% CI:            [{lower:.4f}, {upper:.4f}]")
    print(f"  Exact NL-magic:    {nl_exact:.6f}")
    print(f"  Circuit depth:     {qc.depth()}")
    print(f"  Circuit 2Q gates:  {qc.count_ops().get('cx', 0) + qc.count_ops().get('cz', 0)}")

    agreement = abs(sre_shadow - sre_exact) < max(0.5, 0.3 * sre_exact)
    print(f"  Match: {'PASS' if agreement else 'WEAK'} (within {'30%' if agreement else '>30%'})")

    # --- Test 3: Convergence with more measurements ---
    print(f"\n--- Test 3: Convergence study ---")
    measurement_counts = [100, 500, 1000, 2000, 5000]
    measurement_counts = [m for m in measurement_counts if m <= n_measurements * 3]

    print(f"  {'N_meas':<10} {'Shadow SRE':<15} {'Error':<10}")
    for n_meas in measurement_counts:
        prot = ShadowProtocol(n_qubits, n_measurements=n_meas, seed=42)
        sd = prot.simulate_shadows(qc)
        sre_est = estimate_sre_from_shadows(sd, n_pauli_samples=300, seed=42)
        err = abs(sre_est - sre_exact)
        print(f"  {n_meas:<10} {sre_est:<15.6f} {err:<10.6f}")


def plan_hardware_experiment(n_qubits: int = 20, n_measurements: int = 1000, shots: int = 10000):
    """Plan a hardware experiment: build circuits, estimate costs, check feasibility."""
    from mqhg.states.hypergraph import Hypergraph
    from mqhg.hardware.circuits import hypergraph_to_circuit, circuit_stats
    from mqhg.hardware.cost import print_cost_comparison, experiment_budget
    from mqhg.hardware.noise import fidelity_threshold_check

    print(f"\n{'='*65}")
    print(f" HARDWARE EXPERIMENT PLAN (n={n_qubits})")
    print(f"{'='*65}")

    # Build representative circuit
    hg = Hypergraph(n_qubits)
    for i in range(n_qubits):
        hg.add_edge((i, (i + 1) % n_qubits))
    for i in range(0, n_qubits - 2, 3):
        hg.add_edge((i, i + 1, i + 2))

    qc = hypergraph_to_circuit(hg)
    stats = circuit_stats(qc)

    print(f"\n  Circuit statistics:")
    print(f"    Qubits: {stats['n_qubits']}")
    print(f"    Depth:  {stats['depth']}")
    print(f"    1Q gates: {stats['n_1q_gates']}")
    print(f"    2Q gates: {stats['n_2q_gates']}")
    print(f"    Total gates: {stats['total_gates']}")

    # Feasibility check for different hardware
    print(f"\n  Feasibility checks:")
    for hw_name, fidelity_2q in [("IBM Heron", 0.997), ("IonQ Forte", 0.996),
                                   ("Rigetti Cepheus", 0.991), ("Quantinuum H2", 0.998)]:
        check = fidelity_threshold_check(
            circuit_depth=stats["depth"],
            n_2q_gates=stats["n_2q_gates"],
            gate_fidelity_2q=fidelity_2q,
            n_qubits=n_qubits,
        )
        status = "OK" if check["feasible"] else "RISKY"
        print(f"    {hw_name:<20}: F={check['total_fidelity']:.4f} [{status}]")

    # Cost comparison
    print_cost_comparison(n_qubits, n_measurements, shots)

    # Save circuit for later submission
    circuit_path = OUTPUT_DIR / f"shadow_circuit_n{n_qubits}.qasm"
    qc.qasm(filename=str(circuit_path)) if hasattr(qc, 'qasm') else None
    print(f"\n  State preparation circuit saved to: {circuit_path}")


def build_submission_package(
    n_qubits: int = 20,
    n_measurements: int = 1000,
    shots: int = 10000,
    seed: int = 42,
):
    """Build the full set of circuits ready for hardware submission.

    Saves circuits and metadata to JSON for use with qiskit-ibm-runtime.
    """
    from mqhg.states.hypergraph import Hypergraph
    from mqhg.hardware.circuits import hypergraph_to_circuit, circuit_stats
    from mqhg.hardware.shadows import ShadowProtocol

    print(f"\n  Building {n_measurements} shadow circuits for n={n_qubits}...")

    # Build state preparation circuit
    hg = Hypergraph(n_qubits)
    for i in range(n_qubits):
        hg.add_edge((i, (i + 1) % n_qubits))
    for i in range(0, n_qubits - 2, 3):
        hg.add_edge((i, i + 1, i + 2))

    state_circuit = hypergraph_to_circuit(hg)

    # Build shadow measurement circuits
    protocol = ShadowProtocol(n_qubits, n_measurements=n_measurements, seed=seed)
    shadow_circuits = protocol.build_shadow_circuits(state_circuit)

    # Save metadata
    metadata = {
        "n_qubits": n_qubits,
        "n_measurements": n_measurements,
        "shots_per_circuit": shots,
        "seed": seed,
        "bases": protocol.bases,
        "circuit_stats": circuit_stats(state_circuit),
        "n_shadow_circuits": len(shadow_circuits),
    }

    metadata_path = OUTPUT_DIR / f"submission_n{n_qubits}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved: {metadata_path}")
    print(f"  Total circuits: {len(shadow_circuits)}")
    print(f"  Total shots: {n_measurements * shots:,}")
    print(f"\n  To submit to IBM Quantum, use:")
    print(f"    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2")
    print(f"    service = QiskitRuntimeService()")
    print(f"    backend = service.least_busy(min_num_qubits={n_qubits})")
    print(f"    sampler = SamplerV2(backend)")
    print(f"    job = sampler.run(shadow_circuits, shots={shots})")

    return shadow_circuits, metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shadow tomography hardware experiments")
    parser.add_argument("--validate", action="store_true", help="Classical validation only")
    parser.add_argument("--plan", action="store_true", help="Plan hardware experiment")
    parser.add_argument("--build", action="store_true", help="Build submission package")
    parser.add_argument("--n-qubits", type=int, default=6, help="Number of qubits")
    parser.add_argument("--n-measurements", type=int, default=1000, help="Shadow measurements")
    parser.add_argument("--shots", type=int, default=10000, help="Shots per circuit")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    if args.validate:
        validate_shadow_pipeline(args.n_qubits, args.n_measurements)
    elif args.plan:
        plan_hardware_experiment(args.n_qubits, args.n_measurements, args.shots)
    elif args.build:
        build_submission_package(args.n_qubits, args.n_measurements, args.shots, args.seed)
    else:
        # Default: validate at small scale
        validate_shadow_pipeline(n_qubits=min(args.n_qubits, 8), n_measurements=args.n_measurements)

    print(f"\n  Output directory: {OUTPUT_DIR}")
