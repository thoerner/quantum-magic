"""Quantum hardware module: circuit compilation, shadow tomography, and cost estimation."""

from .circuits import hypergraph_to_circuit, graph_state_circuit, excitation_circuit
from .shadows import ShadowProtocol
from .sre_estimator import estimate_sre_from_shadows, estimate_nonlocal_magic_from_shadows
from .cost import experiment_budget, print_cost_comparison
from .noise import depolarizing_correction, fidelity_threshold_check
