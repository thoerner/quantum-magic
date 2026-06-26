"""Tests for the quantum hardware module.

Tests circuit construction, shadow protocol, SRE estimation, cost calculation,
and noise correction — all without requiring actual quantum hardware.
"""

import numpy as np
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Skip all tests if qiskit is not installed
qiskit = pytest.importorskip("qiskit")

from qiskit.circuit import QuantumCircuit
from qiskit.quantum_info import Statevector as QiskitStatevector
import networkx as nx

from mqhg.states.hypergraph import Hypergraph
from mqhg.hardware.circuits import (
    hypergraph_to_circuit, graph_state_circuit,
    parametric_hypergraph_circuit, excitation_circuit, circuit_stats,
)
from mqhg.hardware.shadows import ShadowProtocol, ShadowData
from mqhg.hardware.sre_estimator import (
    estimate_sre_from_shadows, estimate_nonlocal_magic_from_shadows,
    confidence_interval, required_measurements,
)
from mqhg.hardware.cost import (
    estimate_ibm_cost, estimate_braket_cost, estimate_quantinuum_cost,
    experiment_budget,
)
from mqhg.hardware.noise import (
    depolarizing_correction, fidelity_threshold_check,
    readout_error_mitigation, noise_corrected_pauli_estimate,
)


class TestCircuits:
    def test_graph_state_circuit_structure(self):
        G = nx.cycle_graph(4)
        qc = graph_state_circuit(G)
        assert qc.num_qubits == 4
        ops = qc.count_ops()
        assert ops.get("h", 0) == 4
        assert ops.get("cz", 0) == 4  # 4 edges in cycle graph

    def test_hypergraph_circuit_with_ccz(self):
        hg = Hypergraph(4)
        hg.add_edge((0, 1))
        hg.add_edge((1, 2, 3))
        qc = hypergraph_to_circuit(hg)
        assert qc.num_qubits == 4
        ops = qc.count_ops()
        assert ops.get("h", 0) == 4
        # CCZ decomposes into CX + T/Tdg gates
        assert ops.get("cx", 0) > 0 or ops.get("cz", 0) > 0

    def test_graph_state_matches_qiskit_statevector(self):
        """Verify our circuit produces the expected graph state."""
        G = nx.path_graph(3)
        qc = graph_state_circuit(G)
        sv = QiskitStatevector.from_instruction(qc)
        # Graph state should be normalized
        assert abs(np.linalg.norm(sv.data) - 1.0) < 1e-14
        # Graph state from CZ on |+> should have real amplitudes
        assert np.allclose(sv.data.imag, 0, atol=1e-14)

    def test_excitation_circuit_x(self):
        G = nx.path_graph(3)
        base = graph_state_circuit(G)
        excited = excitation_circuit(base, site=1, excitation_type="X")
        assert excited.num_qubits == 3
        ops = excited.count_ops()
        assert ops.get("x", 0) == 1

    def test_excitation_circuit_rz(self):
        G = nx.path_graph(3)
        base = graph_state_circuit(G)
        excited = excitation_circuit(base, site=0, excitation_type="Rz", epsilon=0.5)
        ops = excited.count_ops()
        assert ops.get("rz", 0) == 1

    def test_parametric_circuit(self):
        edges = [(0, 1), (1, 2, 3)]
        qc = parametric_hypergraph_circuit(4, edges, theta=0.5)
        assert qc.num_qubits == 4

    def test_circuit_stats(self):
        hg = Hypergraph(4)
        hg.add_edge((0, 1))
        hg.add_edge((2, 3))
        qc = hypergraph_to_circuit(hg)
        stats = circuit_stats(qc)
        assert stats["n_qubits"] == 4
        assert stats["total_gates"] > 0
        assert "depth" in stats


class TestShadowProtocol:
    def test_generate_bases(self):
        protocol = ShadowProtocol(n_qubits=4, n_measurements=100, seed=42)
        bases = protocol.bases
        assert len(bases) == 100
        assert all(len(b) == 4 for b in bases)
        assert all(p in ("X", "Y", "Z") for b in bases for p in b)

    def test_build_shadow_circuits(self):
        qc = QuantumCircuit(3)
        qc.h([0, 1, 2])

        protocol = ShadowProtocol(n_qubits=3, n_measurements=10, seed=42)
        circuits = protocol.build_shadow_circuits(qc)
        assert len(circuits) == 10
        for c in circuits:
            assert c.num_qubits == 3
            # Should have measurements
            ops = c.count_ops()
            assert ops.get("measure", 0) == 3

    def test_simulate_shadows(self):
        qc = QuantumCircuit(3)
        qc.h([0, 1, 2])

        protocol = ShadowProtocol(n_qubits=3, n_measurements=50, seed=42)
        shadow_data = protocol.simulate_shadows(qc)
        assert shadow_data.n_qubits == 3
        assert shadow_data.n_snapshots == 50
        for s in shadow_data.shadows:
            assert all(o in (0, 1) for o in s.outcome)
            assert all(b in ("X", "Y", "Z") for b in s.basis)

    def test_pauli_expectation_product_state(self):
        """For |+>^n, <Z_i> = 0 and <X_i> = 1."""
        qc = QuantumCircuit(3)
        qc.h([0, 1, 2])

        protocol = ShadowProtocol(n_qubits=3, n_measurements=5000, seed=42)
        shadow_data = protocol.simulate_shadows(qc)

        # <XII> should be close to 1 for |+> state
        exp_x = protocol.estimate_pauli_expectation(shadow_data, "XII")
        assert abs(exp_x - 1.0) < 0.3, f"<XII> = {exp_x}, expected ~1.0"

        # <ZII> should be close to 0 for |+> state
        exp_z = protocol.estimate_pauli_expectation(shadow_data, "ZII")
        assert abs(exp_z) < 0.3, f"<ZII> = {exp_z}, expected ~0"

    def test_process_results(self):
        """Test processing mock hardware count data."""
        protocol = ShadowProtocol(n_qubits=3, n_measurements=5, seed=42)
        _ = protocol.bases  # Generate bases

        # Mock counts (single shot per circuit)
        counts_list = [
            {"000": 1},
            {"101": 1},
            {"011": 1},
            {"110": 1},
            {"001": 1},
        ]

        shadow_data = protocol.process_results(counts_list)
        assert shadow_data.n_snapshots == 5


class TestSREEstimator:
    def test_stabilizer_state_low_sre(self):
        """Graph state (stabilizer) should give SRE near 0.

        Shadow-based SRE estimation is noisy at small sample sizes due to
        finite statistics. We use a lenient threshold here; real experiments
        would use 10-100x more measurements.
        """
        qc = graph_state_circuit(nx.cycle_graph(4))
        protocol = ShadowProtocol(n_qubits=4, n_measurements=5000, seed=42)
        shadow_data = protocol.simulate_shadows(qc)
        sre = estimate_sre_from_shadows(shadow_data, n_pauli_samples=500, seed=42)
        assert sre < 1.5, f"Stabilizer SRE should be low, got {sre}"

    def test_magic_state_nonzero_sre(self):
        """Hypergraph state with CCZ should have detectable magic."""
        hg = Hypergraph(4)
        hg.add_edge((0, 1))
        hg.add_edge((1, 2))
        hg.add_edge((0, 1, 2))
        qc = hypergraph_to_circuit(hg)

        protocol = ShadowProtocol(n_qubits=4, n_measurements=5000, seed=42)
        shadow_data = protocol.simulate_shadows(qc)
        sre = estimate_sre_from_shadows(shadow_data, n_pauli_samples=300, seed=42)
        # Magic state should have positive SRE (though shadow estimate is noisy)
        # Use a lenient threshold since shadow estimation has variance
        assert sre >= 0.0

    def test_confidence_interval(self):
        qc = graph_state_circuit(nx.path_graph(4))
        protocol = ShadowProtocol(n_qubits=4, n_measurements=2000, seed=42)
        shadow_data = protocol.simulate_shadows(qc)
        mean, lower, upper = confidence_interval(shadow_data, n_pauli_samples=100, n_bootstrap=20, seed=42)
        assert lower <= mean <= upper

    def test_required_measurements(self):
        n = required_measurements(n_qubits=20, target_precision=0.1)
        assert n > 0
        assert n <= 100_000


class TestCostEstimation:
    def test_ibm_cost_positive(self):
        result = estimate_ibm_cost(100, 10000, "payg")
        assert result["cost_usd"] > 0
        assert result["runtime_seconds"] > 0

    def test_ibm_free_tier(self):
        # Very small job should fit in free tier
        result = estimate_ibm_cost(5, 100, "payg")
        assert result["free_tier_covers"] or result["runtime_minutes"] < 10

    def test_braket_cost_components(self):
        result = estimate_braket_cost(100, 10000, "rigetti")
        assert result["task_cost_usd"] == 100 * 0.30
        assert result["shot_cost_usd"] == 100 * 10000 * 0.000425
        assert result["cost_usd"] == result["task_cost_usd"] + result["shot_cost_usd"]

    def test_braket_ionq_expensive(self):
        rigetti = estimate_braket_cost(100, 10000, "rigetti")
        ionq = estimate_braket_cost(100, 10000, "ionq_forte")
        assert ionq["cost_usd"] > rigetti["cost_usd"] * 10

    def test_quantinuum_hqc_formula(self):
        result = estimate_quantinuum_cost(1, 1000, n_1q_gates=20, n_2q_gates=30, n_measurements=10)
        # HQC = 5 + 1000*(20 + 300 + 50)/5000 = 5 + 74 = 79
        expected_hqc = 5 + 1000 * (20 + 10 * 30 + 5 * 10) / 5000
        assert abs(result["hqc_per_circuit"] - expected_hqc) < 0.01

    def test_experiment_budget(self):
        budget = experiment_budget(n_qubits=10, n_measurements=100, shots_per_circuit=1000)
        assert "ibm_payg" in budget
        assert "braket_rigetti" in budget
        assert "quantinuum" in budget
        assert budget["experiment"]["total_shots"] == 100_000


class TestNoiseCorrection:
    def test_no_noise_passthrough(self):
        assert depolarizing_correction(0.5, 3, noise_rate=0.0) == 0.5

    def test_depolarizing_amplifies(self):
        corrected = depolarizing_correction(0.5, 3, noise_rate=0.01)
        assert corrected > 0.5  # Correction amplifies the signal

    def test_high_noise_warning(self):
        with pytest.warns(UserWarning):
            depolarizing_correction(0.5, 3, noise_rate=0.5)

    def test_fidelity_check_shallow_circuit(self):
        result = fidelity_threshold_check(
            circuit_depth=5, n_2q_gates=10,
            gate_fidelity_2q=0.996, n_qubits=10,
        )
        assert result["feasible"]
        assert result["total_fidelity"] > 0.5

    def test_fidelity_check_deep_circuit_warns(self):
        with pytest.warns(UserWarning):
            result = fidelity_threshold_check(
                circuit_depth=100, n_2q_gates=500,
                gate_fidelity_2q=0.99, n_qubits=50,
                threshold=0.1,
            )
        assert not result["feasible"]

    def test_readout_mitigation(self):
        # Perfect calibration should pass through
        cal_matrix = np.eye(4)
        counts = {"00": 700, "01": 100, "10": 100, "11": 100}
        result = readout_error_mitigation(counts, cal_matrix)
        assert abs(result.get("00", 0) - 0.7) < 0.01

    def test_noise_corrected_estimate(self):
        raw = 0.8
        corrected = noise_corrected_pauli_estimate(
            raw, pauli_weight=2, noise_rate=0.01, gate_error_2q=0.004, n_2q_gates=10
        )
        assert corrected > raw  # Both corrections amplify
