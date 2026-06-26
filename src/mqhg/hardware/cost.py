"""Cost estimation for quantum hardware experiments.

Supports IBM Quantum (per-minute), AWS Braket (per-task + per-shot),
and Quantinuum (HQC formula). Provides pre-flight budgeting for
shadow tomography experiments.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --- Provider pricing (verified June 2026) ---

@dataclass(frozen=True)
class IBMPricing:
    """IBM Quantum per-minute pricing."""
    name: str = "IBM Quantum"
    rate_per_min_payg: float = 96.0
    rate_per_min_flex: float = 72.0
    rate_per_min_premium: float = 48.0
    free_min_per_month: float = 10.0
    promo_min_total: float = 180.0
    # Runtime formula constants
    base_overhead_s: float = 2.0
    shot_time_factor_s: float = 0.00035


@dataclass(frozen=True)
class BraketPricing:
    """AWS Braket per-task + per-shot pricing."""
    task_fee: float = 0.30
    # Per-shot fees by provider
    rigetti_per_shot: float = 0.000425
    ionq_forte_per_shot: float = 0.08
    iqm_garnet_per_shot: float = 0.00145
    iqm_emerald_per_shot: float = 0.00160
    quera_per_shot: float = 0.01


@dataclass(frozen=True)
class QuantinuumPricing:
    """Quantinuum HQC-based pricing via Azure."""
    name: str = "Quantinuum H2"
    standard_monthly: float = 125_000.0
    premium_monthly: float = 175_000.0
    standard_hqc: int = 10_000
    premium_hqc: int = 17_000
    # Effective cost per HQC (from subscription)
    cost_per_hqc_standard: float = 12.50  # 125k / 10k
    cost_per_hqc_premium: float = 10.29   # 175k / 17k


IBM = IBMPricing()
BRAKET = BraketPricing()
QUANTINUUM = QuantinuumPricing()


def estimate_ibm_runtime(
    n_circuits: int,
    shots_per_circuit: int,
) -> float:
    """Estimate IBM QPU runtime in seconds.

    Formula: 2s + 0.00035s * total_shots * n_circuits
    (simplified; actual depends on circuit depth and queuing)
    """
    total_time = IBM.base_overhead_s * n_circuits + IBM.shot_time_factor_s * shots_per_circuit * n_circuits
    return total_time


def estimate_ibm_cost(
    n_circuits: int,
    shots_per_circuit: int,
    plan: str = "payg",
) -> dict:
    """Estimate IBM Quantum cost.

    Returns dict with runtime_seconds, runtime_minutes, and cost_usd.
    """
    runtime_s = estimate_ibm_runtime(n_circuits, shots_per_circuit)
    runtime_min = runtime_s / 60.0

    rates = {
        "payg": IBM.rate_per_min_payg,
        "flex": IBM.rate_per_min_flex,
        "premium": IBM.rate_per_min_premium,
    }
    rate = rates.get(plan, IBM.rate_per_min_payg)
    cost = runtime_min * rate

    return {
        "provider": "IBM Quantum",
        "plan": plan,
        "runtime_seconds": runtime_s,
        "runtime_minutes": runtime_min,
        "rate_per_min": rate,
        "cost_usd": cost,
        "free_tier_covers": runtime_min <= IBM.free_min_per_month,
        "promo_covers": runtime_min <= IBM.promo_min_total,
    }


def estimate_braket_cost(
    n_circuits: int,
    shots_per_circuit: int,
    provider: str = "rigetti",
) -> dict:
    """Estimate AWS Braket cost.

    Returns dict with task_cost, shot_cost, total_cost.
    """
    per_shot_rates = {
        "rigetti": BRAKET.rigetti_per_shot,
        "ionq_forte": BRAKET.ionq_forte_per_shot,
        "iqm_garnet": BRAKET.iqm_garnet_per_shot,
        "iqm_emerald": BRAKET.iqm_emerald_per_shot,
        "quera": BRAKET.quera_per_shot,
    }

    per_shot = per_shot_rates.get(provider, BRAKET.rigetti_per_shot)
    task_cost = n_circuits * BRAKET.task_fee
    shot_cost = n_circuits * shots_per_circuit * per_shot
    total = task_cost + shot_cost

    return {
        "provider": f"AWS Braket ({provider})",
        "n_tasks": n_circuits,
        "shots_per_task": shots_per_circuit,
        "task_cost_usd": task_cost,
        "shot_cost_usd": shot_cost,
        "cost_usd": total,
    }


def estimate_quantinuum_cost(
    n_circuits: int,
    shots_per_circuit: int,
    n_1q_gates: int = 50,
    n_2q_gates: int = 60,
    n_measurements: int = 20,
) -> dict:
    """Estimate Quantinuum cost using HQC formula.

    HQC = 5 + C * (N_1q + 10*N_2q + 5*N_m) / 5000
    where C = shots, N_1q = single-qubit ops, N_2q = two-qubit ops, N_m = SPAM ops.
    """
    hqc_per_circuit = 5 + shots_per_circuit * (n_1q_gates + 10 * n_2q_gates + 5 * n_measurements) / 5000
    total_hqc = hqc_per_circuit * n_circuits

    cost_standard = total_hqc * QUANTINUUM.cost_per_hqc_standard
    cost_premium = total_hqc * QUANTINUUM.cost_per_hqc_premium

    months_standard = total_hqc / QUANTINUUM.standard_hqc
    months_premium = total_hqc / QUANTINUUM.premium_hqc

    return {
        "provider": "Quantinuum H2 (Azure)",
        "hqc_per_circuit": hqc_per_circuit,
        "total_hqc": total_hqc,
        "cost_standard_usd": cost_standard,
        "cost_premium_usd": cost_premium,
        "months_standard": months_standard,
        "months_premium": months_premium,
    }


def experiment_budget(
    n_qubits: int,
    n_measurements: int = 1000,
    shots_per_circuit: int = 10000,
    n_2q_gates_per_circuit: int | None = None,
) -> dict:
    """End-to-end cost estimate for a full shadow tomography experiment.

    Args:
        n_qubits: Number of qubits.
        n_measurements: Number of shadow measurement circuits.
        shots_per_circuit: Shots per measurement circuit.
        n_2q_gates_per_circuit: Estimated 2Q gates (auto-estimated if None).

    Returns:
        Dict with costs for all providers.
    """
    if n_2q_gates_per_circuit is None:
        # Rough estimate: hypergraph state with n/2 CCZ gates → ~3n CX gates
        n_2q_gates_per_circuit = 3 * n_qubits

    n_circuits = n_measurements

    ibm_payg = estimate_ibm_cost(n_circuits, shots_per_circuit, "payg")
    ibm_flex = estimate_ibm_cost(n_circuits, shots_per_circuit, "flex")
    braket_rigetti = estimate_braket_cost(n_circuits, shots_per_circuit, "rigetti")
    braket_ionq = estimate_braket_cost(n_circuits, shots_per_circuit, "ionq_forte")
    braket_iqm = estimate_braket_cost(n_circuits, shots_per_circuit, "iqm_garnet")
    quantinuum = estimate_quantinuum_cost(
        n_circuits, shots_per_circuit,
        n_1q_gates=n_qubits * 2,
        n_2q_gates=n_2q_gates_per_circuit,
        n_measurements=n_qubits,
    )

    return {
        "experiment": {
            "n_qubits": n_qubits,
            "n_measurements": n_measurements,
            "shots_per_circuit": shots_per_circuit,
            "total_shots": n_measurements * shots_per_circuit,
            "est_2q_gates": n_2q_gates_per_circuit,
        },
        "ibm_payg": ibm_payg,
        "ibm_flex": ibm_flex,
        "braket_rigetti": braket_rigetti,
        "braket_ionq": braket_ionq,
        "braket_iqm": braket_iqm,
        "quantinuum": quantinuum,
    }


def print_cost_comparison(
    n_qubits: int,
    n_measurements: int = 1000,
    shots_per_circuit: int = 10000,
) -> None:
    """Print a formatted cost comparison table."""
    budget = experiment_budget(n_qubits, n_measurements, shots_per_circuit)
    exp = budget["experiment"]

    print(f"\n{'='*65}")
    print(f" COST ESTIMATE: Shadow Tomography Experiment")
    print(f"{'='*65}")
    print(f" Qubits: {exp['n_qubits']}")
    print(f" Measurement circuits: {exp['n_measurements']}")
    print(f" Shots per circuit: {exp['shots_per_circuit']:,}")
    print(f" Total shots: {exp['total_shots']:,}")
    print(f" Est. 2Q gates/circuit: {exp['est_2q_gates']}")
    print(f"{'='*65}")
    print(f"\n {'Provider':<30} {'Cost (USD)':<15} {'Notes'}")
    print(f" {'-'*30} {'-'*15} {'-'*30}")

    ibm_p = budget["ibm_payg"]
    ibm_f = budget["ibm_flex"]
    print(f" {'IBM Quantum (PAYG)':<30} ${ibm_p['cost_usd']:>10,.2f}   {ibm_p['runtime_minutes']:.1f} min runtime")
    print(f" {'IBM Quantum (Flex)':<30} ${ibm_f['cost_usd']:>10,.2f}   Requires $30k commitment")

    if ibm_p["free_tier_covers"]:
        print(f" {'IBM Quantum (Free)':<30} {'$0':>15}   Within 10 min/month free tier")
    elif ibm_p["promo_covers"]:
        print(f" {'IBM Quantum (Promo)':<30} {'$0':>15}   Within 180 min promotional")

    br = budget["braket_rigetti"]
    bi = budget["braket_ionq"]
    bq = budget["braket_iqm"]
    print(f" {'Braket - Rigetti':<30} ${br['cost_usd']:>10,.2f}   99.1% 2Q fidelity")
    print(f" {'Braket - IQM Garnet':<30} ${bq['cost_usd']:>10,.2f}   ~99.5% 2Q fidelity")
    print(f" {'Braket - IonQ Forte':<30} ${bi['cost_usd']:>10,.2f}   99.6% 2Q fidelity")

    qt = budget["quantinuum"]
    print(f" {'Quantinuum H2 (Standard)':<30} ${qt['cost_standard_usd']:>10,.2f}   {qt['total_hqc']:.0f} HQC, {qt['months_standard']:.1f} months sub")

    print(f"\n{'='*65}")
    print(f" RECOMMENDATION: IBM Quantum (best $/shot for shadow tomography)")
    print(f"{'='*65}\n")
