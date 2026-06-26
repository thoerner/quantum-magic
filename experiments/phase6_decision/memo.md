# Phase 6: Decision Memo

## Classification

- [ ] **Promising**: Concrete evidence magic controls backreaction-like behavior
- [x] **Interesting but weak**: Correlations exist but no causal role established
- [ ] **Mostly metaphorical**: No robust link beyond known holography papers
- [ ] **False direction**: Magic does not add explanatory power

## Evidence Summary

### For the hypothesis

1. **Monotone response law**: B(θ) is monotonically increasing in non-local magic M across all tested system sizes (n=6,8,10) with 100% monotone segments. Linear fit gives B ≈ 0.28·M − 0.18 at n=8 (R²=0.77).

2. **Structured magic requirement**: Local magic (T gates on product states) produces exactly zero MI deformation. Only multi-body non-Clifford gates (CCZ, k≥3 hyperedges) create geometry. This is a non-trivial prediction confirmed at n=8.

3. **Coherent geometry**: Magic-induced MI structure has coherence ratio 3.02 vs 1.84 for Haar-random states (Test 3, n=8). The geometry is structured, not noise.

4. **Frame independence**: MI deformation is invariant under local Clifford rotations to machine precision (Δ < 10⁻¹⁴). The observable is physical.

5. **Locality**: MI decays with ring distance at n=10 — d=1: 0.34, d=2: 0.06, d=3: 0.006, d=5: ~0. Emergent geometry is approximately local.

6. **Phase diagram**: Growth model identifies a rigid→deformable→chaotic transition at p_magic ≈ 0.00 → 0.15, with peak curvature and smoothness in the deformable regime.

### Against the hypothesis

1. **Response law is moderate, not strong**: R² = 0.77 (linear) and 0.73 (power-law). The relationship is clearly monotone but not tightly fit by a simple law. A truly physical law should have R² > 0.95.

2. **Per-pair MI deformation decays with n**: Total MI-deform grows (1.44 at n=4 → 1.78 at n=12) but per-pair MI-deform decays (0.24 → 0.027). This suggests the effect may be a finite-size artifact that vanishes in the thermodynamic limit.

3. **Coupling is not universal**: Overall CoV = 0.71 across magic sources (Test 6). Local T gates produce zero geometry while CCZ produces strong geometry at similar SRE levels. The k-dependence within multi-body gates is milder (CoV = 0.30) but efficiency drops from k=3 (0.36) to k=5 (0.17).

4. **Response coefficient K decreases with n**: K(6) = 0.35, K(8) = 0.29, K(10) = 0.25. If K → 0 as n → ∞, there is no response law in the continuum limit.

5. **Curvature decays rapidly**: Ollivier-Ricci curvature drops from 0.66 (n=4) to 0.02 (n=12). This is partly due to increasing graph sparsity but raises questions about whether emergent curvature persists at scale.

## Key Findings

### Conjecture A (Entanglement → geometry)
- Result: **Confirmed (trivially)**
- Evidence: Graph states produce MI structure; product states do not. This follows from the MI definition and is not a novel claim.
- Confidence: High (by construction)

### Conjecture B (Stabilizer geometry is rigid)
- Result: **Confirmed**
- Evidence: All stabilizer graph states (linear, ring, complete, star) have SRE = 0 and MI deformation = 0 at n=8. The stabilizer MI matrix is exactly zero (flat geometry).
- Confidence: High (exact numerics)

### Conjecture C (Magic enables backreaction)
- Result: **Partially confirmed**
- Evidence: Multi-body non-Clifford gates create non-zero MI deformation proportional to non-local magic. However, local magic (T gates) does not create geometry even at high SRE. The coupling is structure-dependent, not universal.
- Confidence: Moderate

### Conjecture D (Linear response law)
- Result: **Weakly supported**
- Evidence: B is monotone in M (100% of segments). Linear R² = 0.77 at n=8. Power-law exponent α ≈ 1.2 across all n. K decreases with system size.
- Confidence: Low-to-moderate. The relationship exists but "linear" is an overstatement.

## Falsification Tests

### Test 1: Entanglement-only control
- Result: Deep Clifford circuits (high entanglement, low magic) produce less structured MI than magic states.
- Implication: Entanglement alone is insufficient; magic adds something.

### Test 2: Magic-only control
- Result: Product + T gates (high local magic, zero entanglement) produce zero MI deformation.
- Implication: Magic alone is insufficient. Entanglement structure is required. The hypothesis that magic = energy requires *non-local* magic.

### Test 3: Randomness confound
- Result: Structured magic has coherence 3.02 vs Haar random 1.84.
- Implication: Magic-induced geometry is not just random noise. Structured sources create structured geometry.

### Test 4: Basis dependence
- Result: MI deformation invariant under local Cliffords (Δ < 10⁻¹⁴).
- Implication: The observable is frame-independent. SRE varies under basis change, but MI does not.

### Test 5: Locality recovery
- Result: MI decays exponentially with ring distance at n=10.
- Implication: Emergent geometry is local. This is necessary for a gravity interpretation.

### Test 6: Universality of coupling
- Result: Coupling efficiency varies by source (CoV=0.71). Local T gates: efficiency=0. k=3 CCZ: 0.36. k=5: 0.17.
- Implication: The coupling is NOT universal. Different magic sources couple differently to geometry. This is a significant weakness.

### Test 7: Scaling
- Result: Raw MI-deform grows with n (1.44→1.78). Per-pair MI-deform decays (0.24→0.027). SRE/qubit stabilizes at ~0.49. Curvature decays 0.66→0.02.
- Implication: The total effect grows but the intensive quantity decays. The hypothesis needs a clear scaling prediction to survive.

## Growth Model Phase Diagram

The random circuit growth model (n=8, depth=10, 16 p_magic values × 5 seeds) identifies three regimes:

| Regime | p_magic range | SRE | Smoothness | Curvature |
|--------|-------------|-----|------------|-----------|
| Rigid | 0.00 | 0 | ∞ | 0 |
| Deformable | 0.05 – 0.15 | 1–4 | 1–6 | 0–0.4 |
| Chaotic | > 0.3 | 4–6 | 1–2 | 0.1–0.6 |

The transition from rigid to deformable is sharp (occurs at first T/CCZ gate). The deformable→chaotic boundary is less clear, with high variance across seeds.

## Most Likely Failure Mode Observed

**Non-universality of coupling combined with decaying intensive observables.** The fact that different magic sources (local T vs multi-body CCZ) couple completely differently to geometry, and that per-pair MI deformation decays with system size, suggests this may be a finite-size effect of specific gate structures rather than a fundamental physical law.

## Recommended Next Steps

1. **Investigate the scaling law analytically**: Derive whether MI-deform/pair should decay as 1/n, 1/n², or saturate. If there is a scaling prediction from the model that matches the observed decay, the effect may still be physical.

2. **Separate the universality question**: Instead of claiming all magic is equal, refine the hypothesis to state that *non-local magic from k-body interactions* creates geometry, with coupling strength depending on k. This is a weaker but more defensible claim.

3. **Push to n=14-16 with shadow tomography**: The FWHT-accelerated SRE handles n=14 in 14 seconds. For n≥16, use the classical shadow pipeline (already implemented) to estimate SRE from polynomial-time measurements. This would test scaling in a more convincing regime.

4. **Compute connected correlator distance** (already implemented but unused): This may be more sensitive than MI for detecting geometry changes from local excitations, especially in the large-n regime where pairwise MI becomes degenerate.

## Final Verdict

The MQHG framework produces a genuine, non-trivial correlation between non-local magic and emergent geometric deformation. The key insight — that only multi-body non-Clifford interactions (not local T gates) create geometry — is novel and potentially publishable. However, the relationship is moderate (R²=0.77), the coupling is source-dependent, and the intensive quantities decay with system size. The work is best positioned as a **numerical exploration of magic-geometry correlations in small hypergraph states**, not as evidence for emergent gravity. The growth model phase diagram is the most visually compelling result and could anchor a focused paper on phase transitions in magic-modulated quantum geometry.
