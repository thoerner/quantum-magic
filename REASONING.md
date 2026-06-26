# Key Reasoning and Discoveries

Critical insights discovered during development and experimentation. These document
non-obvious decisions, failed approaches, and the physics reasoning behind the
implementation choices.

---

## 1. Why Stabilizer Rényi Entropy (SRE) Fails for Mixed States

**Problem:** The standard SRE formula M_2 = -log₂[Σ|⟨P⟩|⁴ / 4^n] - n gives wrong
results for mixed states. Specifically, the maximally mixed state I/2 (which IS a
stabilizer state) returns M_2 = 1 instead of 0.

**Root cause:** For I/2, only Tr(I·ρ) = 1 is nonzero among the 4 single-qubit
Paulis. So Σξ² = 1, and M_2 = -log₂(1/4) - 1 = 2 - 1 = 1. But I/2 is in the
convex hull of stabilizer states (it's a mixture of |0⟩ and |1⟩), so its magic
should be 0.

**Impact on non-local magic:** For a pure n-qubit state where each qubit's reduced
state is approximately maximally mixed (typical for entangled states), the naïve
formula gives local_magic ≈ n, which always exceeds total_magic, forcing
nonlocal_magic = max(0, total - local) = 0 for ALL states.

**Fix:** Only compute subsystem SRE for approximately-pure reduced states
(Tr(ρ²) > 1 - ε). For mixed subsystems, set local_magic = 0, since the mixedness
comes from entanglement, not from local non-Clifford operations. This gives
physically correct results:
- Product of T states: local magic = Σ M(T|+⟩) > 0, nonlocal = 0 ✓
- Hypergraph states (CCZ|+⟩^n): local magic = 0 (all qubits mixed), nonlocal = total ✓

**Reference:** `src/mqhg/measures/magic.py`, function `_subsystem_sre()`

---

## 2. Why Local Unitary Excitations Cannot Detect Backreaction

This was the most significant conceptual discovery during experimentation.

### Failed Approach 1: Pauli X excitation with MI-based distance

**Observation:** Applying X_i to any state leaves the MI matrix unchanged.

**Proof:** MI(A:B) = S(A) + S(B) - S(AB). Von Neumann entropy is unitarily
invariant: S(U ρ U†) = S(ρ). For a local unitary U_i acting on qubit i:
- If i ∈ A: S(ρ_A) → S(U_i ρ_A U_i†) = S(ρ_A) (invariant)
- If i ∉ A: U_i commutes with Tr_A, so ρ_A is unchanged

Therefore I(A:B) is invariant under ANY single-qubit unitary, not just Paulis.

### Failed Approach 2: T-gate excitation with correlator-based distance

**Observation:** Defined C(i,j) = Σ_{P,Q ∈ {X,Y,Z}} |⟨P_i Q_j⟩ - ⟨P_i⟩⟨Q_j⟩|²
as a distance proxy. Still gave zero backreaction under T-gate excitation.

**Proof:** Under any single-qubit unitary U_0, the operators {X_0, Y_0, Z_0}
undergo an SO(3) rotation: U†P U = Σ_Q R_{PQ} Q. Therefore:

  Σ_P |⟨P_0 Q_j⟩|² → Σ_P |Σ_R R_{PR} ⟨R_0 Q_j⟩|² = Σ_P |⟨P_0 Q_j⟩|²

The sum over all Pauli components is rotationally invariant (it's the squared
Frobenius norm of the correlation matrix, which is a unitarily invariant quantity).

**Conclusion:** ANY geometry proxy based on "total correlation strength between
sites i and j" (whether MI, correlator norm, or any other symmetric function of
the full correlation matrix) is invariant under local unitaries.

### Why This Matters Physically

In a static quantum state, a local unitary is a **gauge transformation**, not a
physical excitation. It doesn't propagate information to other qubits. True
"backreaction" in the holographic sense requires either:
1. Time evolution under a coupling Hamiltonian (excitation propagates)
2. Non-unitary operations (measurement, projection)
3. Comparing different states with different magic content

We chose option 3: backreaction = geometric deformation from stabilizer baseline.

**Reference:** `src/mqhg/analysis/backreaction.py`

---

## 3. The Correct Definition of Backreaction (Static Setting)

**Key insight:** For graph states on 6 qubits (ring, specifically), the pairwise
mutual information I(i:j) = 0 for ALL qubit pairs. This is because no product of
stabilizer generators restricts to a non-trivial 2-qubit operator on any pair
{i,j} without involving additional qubits.

Proof for ring graph: Each stabilizer K_i = X_i Z_{i-1} Z_{i+1} involves 3 qubits.
Any product of stabilizers either involves X on activated qubits (can't produce
a Z-only 2-qubit operator) or is the identity. Therefore no non-trivial 2-local
Pauli has expectation ±1, implying ρ_{ij} = I/4 (maximally mixed) for all pairs.

**Physical interpretation:** The ring graph state has **flat emergent geometry**.
There is no preferred distance structure — all pairs are equally (un)correlated.

**When CCZ is applied:** The 3-body controlled-phase creates genuine pairwise
correlations (verified: I(0:1) ≈ 0.31, I(1:2) ≈ 0.40 for ring+CCZ at n=6).
Magic **creates geometry from flatness**.

**Correct backreaction definition:**
```
B(ψ) = ||MI(ψ) - MI(ψ_stabilizer)||_F
```

For states derived from a graph state by adding magic phases, the stabilizer
reference has MI = 0 (flat), so B = ||MI(ψ)||_F = the Frobenius norm of the
MI matrix itself.

**Reference:** `src/mqhg/analysis/backreaction.py`, `src/mqhg/models/sandbox.py`

---

## 4. Not All Graph States Have Flat Geometry

**Observation from Conjecture B test:** While ring and similar "closed" graph states
have zero pairwise MI, other graph structures (linear, complete, star) DO have
non-zero MI even as pure stabilizer states.

**Explanation:** For the linear graph state, the endpoint qubit 0 has stabilizer
K_0 = X_0 Z_1 (only 2 qubits). This means ⟨X_0 Z_1⟩ = 1, giving the pair (0,1)
a non-trivial correlation and hence I(0:1) > 0.

**Criterion for flat geometry:** A graph state has I(i:j) = 0 for all pairs if
and only if every stabilizer generator involves ≥3 qubits, which happens when
every vertex has degree ≥ 2 and the graph has no "leaf" structures that isolate
2-qubit correlations.

**Implication:** When comparing magic states to stabilizer references, the reference
MI should be computed from the ACTUAL stabilizer state (at θ=0), not assumed to be
zero. The ring graph is a convenient choice because it gives a truly flat baseline.

---

## 5. Magic and Geometry Are Correlated but Not Identical

**Sweep results (n=6, ring base, 3-body magic phase θ from 0 to π):**
```
θ=0.00: SRE=0.00, BR=0.00   (stabilizer, flat)
θ=1.57: SRE=3.18, BR=0.63   (half-CCZ, emerging geometry)
θ=2.47: SRE=3.69, BR=1.21   (SRE peaks here)
θ=3.14: SRE=3.36, BR=1.44   (full CCZ, max geometry)
```

**Key observations:**
1. SRE peaks before θ=π (around θ≈2.47) then decreases slightly
2. MI deformation (backreaction) increases monotonically all the way to θ=π
3. At θ=π, SRE is NOT at its maximum, but geometry IS at maximum deformation

**Interpretation:** The relationship between magic and geometry is not simply
"more magic = more geometry." The geometric deformation depends on the STRUCTURE
of the magic (which correlations are created by the specific phase angles), not
just the total magic content. This is consistent with the research hypothesis that
non-local magic (not just total magic) is the relevant quantity for geometry.

---

## 6. Reconstruction Error as Holographic Backreaction

**Phase 4 result:** For the approximate holographic code, reconstruction error
increases monotonically with magic strength ε:
```
ε=0.000: err=0.000  (perfect stabilizer code, no backreaction)
ε=1.000: err=0.729  (maximal magic, significant degradation)
ε=2.000: err=0.968  (wrapping around, state becomes nearly orthogonal)
```

**Interpretation:** In the holographic picture, "backreaction" means the code's
ability to protect information degrades. Magic distorts the code structure,
making previously correctable errors un-correctable. This is the tensor-network
analog of "matter curving spacetime" — the information-theoretic code structure
(geometry) changes in response to non-stabilizer content (matter/energy).

This is complementary to the Phase 3 result (magic creates MI structure from
flatness). Together they say:
- Phase 3: Magic creates geometric structure (bulk perspective)
- Phase 4: Magic degrades error correction (boundary/code perspective)

Both are manifestations of the same phenomenon viewed from different sides of
the holographic duality.

---

## 7. The Correlator Distance Function (Retained for Future Use)

The `correlator_distance_matrix()` function in `measures/geometry.py` computes:
```
d(i,j) = -log(√(Σ_{P,Q} |C(P_i,Q_j)|²) + ε)
```

While this is invariant under local unitaries (making it unsuitable for
excitation-based backreaction), it IS a valid distance proxy for comparing
DIFFERENT states — it captures the total correlation strength between qubit
pairs regardless of the local basis choice. It's retained for potential use in:
- Comparing geometry across state families
- Detecting phase transitions in the growth model
- Providing a basis-independent alternative to MI for geometry extraction
