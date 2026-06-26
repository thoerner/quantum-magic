"""State preparation circuit construction targeting Qiskit.

Converts MQHG hypergraph/graph state definitions into Qiskit QuantumCircuit
objects suitable for execution on IBM Quantum hardware.
"""

from __future__ import annotations

from math import pi

import networkx as nx
import numpy as np

try:
    from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister
except ImportError:
    raise ImportError(
        "qiskit is required for the hardware module. "
        "Install with: pip install 'mqhg[hardware]'"
    )

from ..states.hypergraph import Hypergraph


def hypergraph_to_circuit(hypergraph: Hypergraph) -> QuantumCircuit:
    """Convert a Hypergraph to a Qiskit QuantumCircuit.

    Starts with H on all qubits (to prepare |+>^n), then applies:
    - CZ for 2-body edges (native on IBM)
    - Decomposed CCZ for 3-body edges (6 CX + T/Tdg)
    - General multi-controlled phase for higher-order edges
    """
    n = hypergraph.n_qubits
    qc = QuantumCircuit(n, name="hypergraph_state")

    # Prepare |+>^n
    for q in range(n):
        qc.h(q)

    # Apply edge gates
    for edge, phase in zip(hypergraph.edges, hypergraph.phases):
        k = len(edge)
        if k < 2:
            continue

        if k == 2:
            _append_controlled_phase_2q(qc, edge[0], edge[1], phase)
        elif k == 3:
            _append_controlled_phase_3q(qc, edge[0], edge[1], edge[2], phase)
        else:
            _append_multi_controlled_phase(qc, list(edge), phase)

    return qc


def graph_state_circuit(graph: nx.Graph) -> QuantumCircuit:
    """Convert a NetworkX graph to a graph-state preparation circuit.

    All edges are CZ (Clifford, native on IBM hardware).
    """
    n = graph.number_of_nodes()
    qc = QuantumCircuit(n, name="graph_state")

    for q in range(n):
        qc.h(q)

    for i, j in graph.edges():
        qc.cz(i, j)

    return qc


def parametric_hypergraph_circuit(
    n_qubits: int,
    edges: list[tuple[int, ...]],
    theta: float,
) -> QuantumCircuit:
    """Parametric hypergraph state circuit with uniform phase theta.

    |ψ_H(θ)> = ∏_e exp(iθ ∏_{v∈e} Z_v) |+>^⊗n
    Realized as controlled-phase(2θ) gates on each edge.
    """
    hg = Hypergraph(n_qubits)
    for e in edges:
        hg.add_edge(e, 2 * theta)
    return hypergraph_to_circuit(hg)


def excitation_circuit(
    base_circuit: QuantumCircuit,
    site: int,
    excitation_type: str = "X",
    epsilon: float = 0.1,
) -> QuantumCircuit:
    """Append a localized excitation gate to an existing circuit.

    Args:
        base_circuit: The state preparation circuit.
        site: Qubit index to excite.
        excitation_type: "X", "Z", "T", or "Rz" (uses epsilon).
        epsilon: Rotation angle for Rz excitation.

    Returns:
        New circuit with excitation appended.
    """
    qc = base_circuit.copy()
    qc.name = f"{base_circuit.name}_excite_{excitation_type}_{site}"

    if excitation_type == "X":
        qc.x(site)
    elif excitation_type == "Z":
        qc.z(site)
    elif excitation_type == "T":
        qc.t(site)
    elif excitation_type == "Rz":
        qc.rz(epsilon, site)
    else:
        try:
            angle = float(excitation_type)
            qc.p(angle, site)
        except ValueError:
            raise ValueError(f"Unknown excitation type: {excitation_type}")

    return qc


# --- Decomposition helpers ---


def _append_controlled_phase_2q(qc: QuantumCircuit, q0: int, q1: int, phase: float) -> None:
    """Append a two-qubit controlled-phase gate.

    CZ (phase=pi) is native. For other phases, decompose into
    Rz + CX + Rz + CX.
    """
    if np.isclose(phase, pi):
        qc.cz(q0, q1)
    else:
        # cp(phase) = diag(1, 1, 1, e^{i*phase})
        # Decomposition: Rz(phase/2) on both + CX + Rz(-phase/2) on target + CX
        qc.p(phase / 2, q0)
        qc.p(phase / 2, q1)
        qc.cx(q0, q1)
        qc.p(-phase / 2, q1)
        qc.cx(q0, q1)


def _append_controlled_phase_3q(
    qc: QuantumCircuit, q0: int, q1: int, q2: int, phase: float
) -> None:
    """Append a three-qubit controlled-controlled-phase gate.

    For phase=pi (CCZ), uses the standard decomposition into
    6 CX + T/Tdg gates. For general phase, uses a relative-phase
    Toffoli-like decomposition.
    """
    if np.isclose(phase, pi):
        # Standard CCZ decomposition:
        # CCZ = (I⊗I⊗H) · Toffoli · (I⊗I⊗H)
        # Toffoli decomposes into 6 CX + T/Tdg
        _append_ccz(qc, q0, q1, q2)
    else:
        # General CCC-phase: decompose into two controlled-phase gates
        # and one Toffoli-like structure
        half = phase / 2
        _append_controlled_phase_2q(qc, q1, q2, half)
        qc.cx(q0, q1)
        _append_controlled_phase_2q(qc, q1, q2, -half)
        qc.cx(q0, q1)
        _append_controlled_phase_2q(qc, q0, q2, half)


def _append_ccz(qc: QuantumCircuit, q0: int, q1: int, q2: int) -> None:
    """Append CCZ using the standard T-gate decomposition (6 CX + 7 T/Tdg)."""
    qc.cx(q1, q2)
    qc.tdg(q2)
    qc.cx(q0, q2)
    qc.t(q2)
    qc.cx(q1, q2)
    qc.tdg(q2)
    qc.cx(q0, q2)
    qc.t(q1)
    qc.t(q2)
    qc.cx(q0, q1)
    qc.t(q0)
    qc.tdg(q1)
    qc.cx(q0, q1)


def _append_multi_controlled_phase(qc: QuantumCircuit, qubits: list[int], phase: float) -> None:
    """Append a multi-controlled phase gate on k qubits.

    Uses recursive decomposition: reduce k-controlled to (k-1)-controlled
    gates using one ancilla-free decomposition step.
    """
    k = len(qubits)
    if k == 2:
        _append_controlled_phase_2q(qc, qubits[0], qubits[1], phase)
    elif k == 3:
        _append_controlled_phase_3q(qc, qubits[0], qubits[1], qubits[2], phase)
    else:
        # Recursive: split into two halves
        # C^k-Phase ≈ C^(k-1)-Phase(half) + CX + C^(k-1)-Phase(-half) + CX + ...
        half = phase / 2
        _append_multi_controlled_phase(qc, qubits[1:], half)
        qc.cx(qubits[0], qubits[1])
        _append_multi_controlled_phase(qc, qubits[1:], -half)
        qc.cx(qubits[0], qubits[1])
        _append_controlled_phase_2q(qc, qubits[0], qubits[-1], half)


def circuit_stats(qc: QuantumCircuit) -> dict:
    """Extract circuit statistics relevant for cost estimation."""
    ops = qc.count_ops()
    n_1q = sum(v for k, v in ops.items() if k in ("h", "x", "y", "z", "s", "t", "tdg", "rz", "p", "sx"))
    n_2q = sum(v for k, v in ops.items() if k in ("cx", "cz", "ecr", "rzz"))
    return {
        "n_qubits": qc.num_qubits,
        "depth": qc.depth(),
        "n_1q_gates": n_1q,
        "n_2q_gates": n_2q,
        "total_gates": sum(ops.values()),
        "ops": dict(ops),
    }
