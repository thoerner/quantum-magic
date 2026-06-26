# Phase 1: Literature Map

## Format

| Topic | Key Papers | What is Established | What Remains Speculative |
|-------|-----------|--------------------|-----------------------|

## Required Topics

### 1. Stabilizer States and Magic

| Topic | Key Papers | Established | Speculative |
|-------|-----------|-------------|-------------|
| Stabilizer formalism | Gottesman (1997), Aaronson-Gottesman (2004) | Stabilizer states form a discrete subset efficiently simulable classically (Gottesman-Knill). Clifford group preserves stabilizer states. | — |
| Magic states | Bravyi-Kitaev (2005) "Universal quantum computation with ideal Clifford gates and noisy ancillas" | Magic state distillation enables universal QC from Clifford + noisy non-stabilizer ancillas. T-state is canonical magic resource. | Optimal distillation protocols; connection to fault-tolerance thresholds. |

### 2. Stabilizer Rényi Entropy

| Topic | Key Papers | Established | Speculative |
|-------|-----------|-------------|-------------|
| SRE definition | Leone-Oliviero-Hamma (2022) "Stabilizer Rényi entropy" | SRE_α is a computable magic monotone for pure states. Based on flatness of Pauli expectation distribution. Zero iff stabilizer state. | Extension to mixed states; operational interpretation beyond monotone. |
| Computational aspects | Various (2022-2024) | SRE_2 requires O(4^n) Pauli evaluations for exact computation. Efficient estimation via random sampling for large n. | Whether polynomial-time approximation suffices for physics applications. |

### 3. Quantum Hypergraph States

| Topic | Key Papers | Established | Speculative |
|-------|-----------|-------------|-------------|
| Hypergraph states | Rossi et al. (2013), Gühne et al. (2014) | Hypergraph states generalize graph states via multi-qubit controlled-Z gates. k>2 body edges inject non-stabilizerness. | Role in quantum advantage beyond magic resource counting. |
| Magic of hypergraph states | Chen-Yan-Zhou (2024) "Magic of quantum hypergraph states" | Random hypergraph states typically achieve high magic. Degree structure controls magic content. Constant-degree hypergraphs have bounded magic. | Whether structured (non-random) hypergraphs can produce controlled intermediate magic relevant to physics. |

### 4. Holographic Tensor-Network Codes

| Topic | Key Papers | Established | Speculative |
|-------|-----------|-------------|-------------|
| HaPPY code | Pastawski-Yoshida-Harlow-Preskill (2015) "Holographic quantum error-correcting codes" | Perfect tensor networks on hyperbolic tilings reproduce RT formula, subregion duality, and error correction properties of AdS/CFT. | Whether toy models capture dynamical (time-dependent) gravity. |
| Bulk reconstruction | Almheiri-Dong-Harlow (2015) "Bulk locality and quantum error correction in AdS/CFT" | Bulk operators reconstructable from boundary subregions via QEC. Entanglement wedge reconstruction. | State-dependent reconstruction; complexity of reconstruction. |
| RT from QEC | Harlow (2017) "The Ryu-Takayanagi formula from quantum error correction" | RT formula emerges naturally from operator-algebra QEC in holographic codes. | Beyond leading order; quantum corrections. |

### 5. Approximate Holographic Codes

| Topic | Key Papers | Established | Speculative |
|-------|-----------|-------------|-------------|
| Approximate QEC | Various (2019-2023) | Perfect codes too rigid for full CFT physics. Approximate codes allow finite-N effects, correlations, state dependence. | Precise characterization of which approximate codes have geometric duals. |

### 6. Entanglement-First-Law Derivations of Einstein Equations

| Topic | Key Papers | Established | Speculative |
|-------|-----------|-------------|-------------|
| Linearized Einstein | Faulkner-Lewkowycz-Maldacena (2013), Lashkari et al. (2014) | First law of entanglement entropy → linearized Einstein equations in holographic settings. δS = δ<K> relates boundary entanglement to bulk geometry. | Nonlinear regime; non-holographic settings. |
| Quantum corrections | Faulkner-Lewkowycz-Maldacena (2013) "Quantum corrections to holographic entanglement entropy" | 1/N corrections to RT formula include bulk entanglement term. | Higher-order corrections; connection to magic. |

### 7. Gravitational Backreaction and Non-Local Magic

| Topic | Key Papers | Established | Speculative |
|-------|-----------|-------------|-------------|
| Magic-backreaction link | Cao-Cheng-Hamma-Leone-Munizzi-Oliviero (2024) "Gravitational back-reaction is magical" | In holographic CFTs: non-local magic vanishes iff no gravitational backreaction. Non-local magic bounds related to cosmic brane tension response. | Whether this is causal or merely diagnostic; whether it extends beyond AdS/CFT. |

### 8. Tensor Networks and Emergent Geometry

| Topic | Key Papers | Established | Speculative |
|-------|-----------|-------------|-------------|
| Geometry from entanglement | Van Raamsdonk (2010), Swingle (2012), Maldacena-Susskind (2013) | Entanglement structure encodes geometric connectivity. MERA-like tensor networks have hyperbolic geometry. Disconnecting entanglement disconnects spacetime. | Whether entanglement alone determines geometry uniquely; role of dynamics. |

---

## Anchor Papers Checklist

- [ ] Bravyi-Kitaev magic states (2005)
- [ ] Leone-Oliviero-Hamma stabilizer Rényi entropy (2022)
- [ ] Pastawski-Yoshida-Harlow-Preskill HaPPY code (2015)
- [ ] Almheiri-Dong-Harlow bulk locality and QEC (2015)
- [ ] Harlow RT from QEC (2017)
- [ ] Faulkner/Lewkowycz/Maldacena quantum corrections (2013)
- [ ] Cao/Cheng/Hamma/Leone/Munizzi/Oliviero gravitational backreaction is magical (2024)
- [ ] Chen/Yan/Zhou magic of quantum hypergraph states (2024)

## Next Steps

- Deep-read each anchor paper and extract:
  - Precise claims with conditions
  - Mathematical framework used
  - Limitations acknowledged by authors
  - Open questions relevant to our hypothesis
- Identify gaps between papers that this research aims to bridge
