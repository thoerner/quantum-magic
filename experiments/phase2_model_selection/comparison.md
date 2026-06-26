# Phase 2: Formal Model Selection

## Three Representations to Compare

### 1. Hypergraph State Model

| Criterion | Assessment |
|-----------|-----------|
| Represent entanglement geometry? | Yes — mutual information from state defines emergent distances |
| Represent magic? | Yes — directly via hyperedge order (k≥3 → non-Clifford) |
| Represent localized matter excitations? | Yes — local Pauli/phase defects |
| Define backreaction observables? | Yes — ΔD from excitation |
| Scale computationally? | Statevector: exact to n≈14. Stabilizer simulation: exponential in magic |

**Strengths:** Direct connection between hypergraph structure and magic content. Clean parametric control (edge size, phase). Well-studied in quantum information.

**Weaknesses:** No built-in notion of "bulk" vs "boundary". Geometry must be extracted post-hoc rather than being architecturally encoded.

### 2. Factor-Graph Amplitude Model

| Criterion | Assessment |
|-----------|-----------|
| Represent entanglement geometry? | Yes — factor graph structure defines correlation decay |
| Represent magic? | Partially — non-stabilizer factors exist but less studied |
| Represent localized matter excitations? | Yes — local factor perturbation |
| Define backreaction observables? | Yes — Δ correlation structure |
| Scale computationally? | Tensor contraction: depends on treewidth. Approximate: variational methods |

**Strengths:** Natural for statistical mechanics intuition. Factor graph ↔ tensor network correspondence is well-understood. Amenable to belief propagation approximations.

**Weaknesses:** Magic is less directly characterized in this language. The stabilizer/non-stabilizer boundary is cleaner in gate/state language.

### 3. Tensor-Network Holographic Code Model

| Criterion | Assessment |
|-----------|-----------|
| Represent entanglement geometry? | Yes — network geometry IS the emergent geometry (by construction) |
| Represent magic? | Yes — tensor perfectness/stabilizerness is controllable |
| Represent localized matter excitations? | Yes — bulk logical insertions |
| Define backreaction observables? | Yes — min-cut, reconstruction region, boundary entropy |
| Scale computationally? | Small codes: exact. Large: tensor contraction, limited by bond dimension |

**Strengths:** Built-in holographic interpretation. RT formula emerges naturally. Bulk/boundary distinction is architectural. Direct connection to AdS/CFT literature.

**Weaknesses:** Restricted to holographic geometry (hyperbolic). Less general than free hypergraph construction. Perfect tensor requirement is restrictive.

## Recommendation

**Phase 3 (exact numerics): Start with hypergraph states.**
- Simplest implementation
- Direct magic control via edge structure
- Feasible for n=4–12 qubits exactly
- Clean falsification tests

**Phase 4 (holographic): Move to HaPPY-like codes.**
- Adds holographic interpretation
- Tests whether hypergraph results translate to holographic setting
- Connects to AdS/CFT literature

**Phase 5 (response law): Use both in parallel.**
- Hypergraph: general response law derivation
- Holographic: RT-connected response law

## Decision Matrix

| Priority | Model | Phase | Reason |
|----------|-------|-------|--------|
| 1 | Hypergraph states | 3 | Fastest to implement, direct magic control |
| 2 | HaPPY code | 4 | Holographic interpretation |
| 3 | Approximate code | 4-5 | Bridges rigid/deformable |
| 4 | Factor graph | 5+ | Statistical mechanics connection (optional) |
