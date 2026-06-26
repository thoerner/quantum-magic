# Phase 5: Candidate Response Law Derivation

## Target Form

We seek an effective equation:

$$\frac{\delta A_\gamma}{4 G_{\text{eff}}} + \delta S_{\text{bulk}} \approx \delta \langle K_A \rangle + \text{correction}_{\text{magic}}$$

or equivalently:

$$\delta \text{Geometry} = \text{Entanglement response} + \text{Magic response}$$

## Starting Point: Entanglement First Law

In holographic settings, the entanglement first law gives:

$$\delta S_A = \delta \langle K_A \rangle$$

where $K_A$ is the modular Hamiltonian of region $A$.

Combined with the RT formula $S_A = A_\gamma / 4G_N + S_\text{bulk}$:

$$\frac{\delta A_\gamma}{4G_N} + \delta S_\text{bulk} = \delta \langle K_A \rangle$$

This gives linearized Einstein equations when varied over all regions $A$.

## The Magic Correction

**Hypothesis:** In non-stabilizer codes, the RT formula receives a correction:

$$S_A = \frac{A_\gamma}{4G_N} + S_\text{bulk} + f(M_\text{nonlocal})$$

where $f$ is a function of non-local magic that:
- Vanishes for stabilizer states (perfect codes)
- Is positive and bounded for physical states
- Controls the state-dependence of the entanglement wedge

## Derivation Attempt

### Step 1: Define the observables in the toy model

For a hypergraph state $|\psi_H\rangle$:

- **Geometry:** $D_{ij} = -\xi \log(I(i:j) + \epsilon)$
- **Excitation energy:** $\delta E_i = \langle \psi'|H_\text{local}|\psi'\rangle - \langle \psi|H_\text{local}|\psi\rangle$  
- **Backreaction:** $B_i = ||D' - D||_F$ restricted to neighborhood of $i$

### Step 2: Linear response ansatz

For small perturbations:

$$B_i(\epsilon) \approx K_i \cdot \epsilon$$

where $K_i$ is the response coefficient at site $i$.

**Conjecture:** $K_i$ is monotone in non-local magic:

$$K_i(|\psi_\text{stab}\rangle) \approx 0$$
$$K_i(|\psi_\text{magic}\rangle) > 0$$
$$K_i \propto M_\text{nonlocal}$$  (strong version)

### Step 3: Numerical test plan

1. For each state class, compute $K_i$ via linear regression of $B_i(\epsilon)$ over small $\epsilon$.
2. Plot $K_i$ vs $M_\text{nonlocal}$ across state families.
3. Test whether the relation is:
   - Linear (strong success)
   - Monotone (medium success)  
   - Uncorrelated (failure)

### Step 4: Connection to holographic setting

In the HaPPY code:
- $A_\gamma$ → min-cut through tensor network
- $S_\text{bulk}$ → entropy of bulk logical state
- $M_\text{nonlocal}$ → non-local magic of the encoded state

**Test:** Does $\delta(\text{min-cut})$ correlate with $M_\text{nonlocal}$ when bulk state is perturbed?

## Assumptions Made Explicit

1. Mutual information defines a meaningful emergent distance (not proven for general states)
2. The Frobenius norm of distance change is a good backreaction proxy (other norms may differ)
3. Non-local magic is the relevant quantity, not total magic (motivated by Cao et al.)
4. Linear response regime exists (may break for large perturbations or highly chaotic states)
5. The toy model captures essential features of holographic backreaction (big assumption)

## Success/Failure Criteria for This Phase

**Success:** Find $K_i \propto M_\text{nonlocal}$ with $R^2 > 0.5$ across multiple state families.

**Partial success:** Monotone relationship without clean proportionality.

**Failure:** No correlation, or correlation driven entirely by entanglement (not magic specifically).
