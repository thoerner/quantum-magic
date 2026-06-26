# MQHG: Magic-Modulated Quantum Hypergraph Models of Emergent Gravity

Research codebase investigating whether gravitational backreaction can be modeled as coarse-grained dynamics of a quantum computational hypergraph, where entanglement supplies emergent spatial geometry and non-stabilizer "magic" supplies the deformability needed for gravitational backreaction.

## Core Hypothesis

> A stabilizer-like holographic code gives inert emergent space. Non-stabilizer (magic-bearing) structure makes the encoding imperfect. Controlled imperfection allows matter and geometry to couple. That coupling appears macroscopically as gravitational backreaction.

## Installation

```bash
pip install -e ".[dev]"
```

Requires Python ≥ 3.10. Dependencies: NumPy, SciPy, NetworkX, Matplotlib, JAX.

## Project Structure

```
src/mqhg/
├── core/           # Statevector simulation, quantum gates
├── states/         # State preparation (hypergraph, graph, random circuits)
├── measures/       # Entanglement, magic (SRE), and emergent geometry measures
├── models/         # Toy models (sandbox, HaPPY code, approximate code, growth)
└── analysis/       # Backreaction observables and plotting

experiments/
├── phase1_literature/      # Literature map
├── phase2_model_selection/ # Model comparison and ranking
├── phase3_exact_numerics/  # Small-system exact simulations
├── phase4_holographic/     # HaPPY-code and holographic experiments
├── phase5_response_law/    # Candidate response law derivation
└── phase6_decision/        # Final verdict memo

tests/              # Unit tests (pytest)
```

## Running Tests

```bash
python3 -m pytest tests/ -v
```

## Running Experiments

### Phase 3: Exact Numerics (Hypergraph Sandbox)

```bash
python3 experiments/phase3_exact_numerics/run_sandbox.py [n_qubits]
```

Tests Conjectures B and C: whether stabilizer states are geometrically rigid and whether magic enables backreaction.

### Phase 4: Holographic Codes

```bash
python3 experiments/phase4_holographic/run_happy.py
```

Compares stabilizer vs non-stabilizer HaPPY-like codes.

## Key Quantities

| Category | Measures |
|----------|----------|
| Entanglement | von Neumann entropy, Rényi entropy, mutual information, entanglement spectrum |
| Magic | Stabilizer Rényi entropy (SRE₂), mana, non-local magic |
| Geometry | MI-distance matrix, emergent metric, Ollivier-Ricci curvature, spectral dimension |
| Backreaction | ΔD (distance matrix change), local/global deformation, response coefficient |

## Research Phases

1. **Literature map** — Anchor papers and established vs speculative claims
2. **Model selection** — Hypergraph states vs factor graphs vs tensor-network codes
3. **Exact numerics** — Statevector simulations for n=4–14 qubits
4. **Holographic codes** — HaPPY code perturbation and approximate codes
5. **Response law** — Derive candidate δGeometry = K × δEnergy relation
6. **Decision memo** — Classify direction as promising/weak/metaphorical/false

## Computational Limits

- SRE computation is O(4^n) — feasible up to ~12 qubits exactly
- Statevector simulation is O(2^n) memory — feasible up to ~14 qubits
- Geometry measures (Ollivier-Ricci curvature) involve LP solving per edge
