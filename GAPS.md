# Known Gaps and Open Work

Tracked gaps between the current implementation and the full research spec.

## High-Impact (Blocks Real Results)

### 1. Falsification Tests 2–7 Not Implemented

Only Test 1 (entanglement-only control) has a runner in `run_sandbox.py`. The remaining six tests from Section 10 of the spec have no code:

- **Test 2**: Magic-only control — high magic, weak/scrambled entanglement geometry
- **Test 3**: Randomness confound — distinguish controlled non-local magic from generic chaos
- **Test 4**: Basis dependence — verify results are frame-independent
- **Test 5**: Locality recovery — confirm emergent dynamics become approximately local after coarse-graining
- **Test 6**: Universality of coupling — all excitation types should curve geometry equally
- **Test 7**: Scaling — verify effects persist (or grow) with system size n

Location: `experiments/phase3_exact_numerics/run_sandbox.py`

### 2. Phase 5 Response Law Has No Numerics

`experiments/phase5_response_law/derivation.md` contains the mathematical framework but no script that:

- Computes K_i (response coefficient) via linear regression of B_i(ε) over small ε
- Plots K_i vs M_nonlocal across state families
- Reports R² and determines if the relationship is linear, monotone, or absent

Needs: `experiments/phase5_response_law/run_response_law.py`

### 3. HaPPY Code Is Too Minimal

Current implementation (`src/mqhg/models/happy.py`):
- Only [[5,1,3]] perfect code (1 logical → 5 physical)
- No multi-layer hyperbolic tiling
- No proper min-cut via tensor network graph structure
- No entanglement wedge reconstruction

The spec calls for:
- Multiple bulk logical qubits
- Pentagon tiling with ≥2 layers
- Min-cut computation from network topology
- Entanglement wedge boundary tracking under perturbation

### 4. ~~No Experiments Have Been Run~~ (RESOLVED)

Phase 3 and Phase 4 experiments have been run at n=6. Key results documented in
`REASONING.md`. Remaining work: scale to larger n, run falsification tests 2–7.

## Medium-Impact (Limits Quality or Scale)

### 5. No JAX Acceleration

JAX is a listed dependency but nothing uses it. Targets for JIT compilation:
- `measures/magic.py`: O(4^n) Pauli expectation loop in `stabilizer_renyi_entropy()`
- `measures/entanglement.py`: O(n²) pairwise `mutual_information_matrix()`
- `measures/geometry.py`: Ollivier-Ricci curvature LP per edge

Would extend feasible exact SRE from n≈12 to n≈14-15.

### 6. No Data Persistence

Experiment results print to stdout only. No JSON/HDF5 serialization for:
- Sweep results (magic vs backreaction data points)
- Shadow tomography raw data
- Cross-experiment comparison

### 7. Approximate Holographic Code Encoder Is Ad-Hoc

`src/mqhg/models/approximate.py`: the `_clifford_encoder()` method does not produce a known stabilizer code. It applies Hadamards + a trivial structure. Should use a proper code construction (e.g., random stabilizer code from tableau, or a known small code family).

### 8. ~~Non-Local Magic for Mixed States Is a Proxy~~ (RESOLVED)

Fixed: `_subsystem_sre()` now only computes SRE for approximately-pure subsystems.
For mixed subsystems (from entanglement), local magic = 0. This avoids the
fundamental issue that the SRE formula conflates mixedness with magic for non-pure
states (see `REASONING.md` §1 for full derivation). The shadow-based version in
`hardware/sre_estimator.py` still uses the Pauli-expectation proxy but operates on
the global pure state, not mixed subsystems.

## Lower-Priority (Nice to Have)

### 9. No Jupyter Notebooks

`notebooks/` directory is empty. Interactive exploration notebooks would help with:
- Visualizing geometry deformation
- Parameter sweeps with inline plots
- Comparing state families side by side

### 10. No Tensor Network Contraction Path

Only statevector simulation exists. For larger approximate holographic codes (n > 14), tensor contraction via `opt_einsum` or a dedicated TN library (quimb, tensornetwork) would extend computational reach without quantum hardware.

### 11. Growth Model Phase Diagram Untested

`RandomCircuitGrowth.phase_diagram()` in `src/mqhg/models/growth.py` is implemented but the predicted three-phase structure (rigid → deformable → chaotic) has not been validated numerically.

### 12. Literature Map Is Skeleton Only

`experiments/phase1_literature/literature_map.md` lists paper names and topics but contains no extracted claims, precise conditions, or gap analysis from deep reading.

## Suggested Priority Order

1. Run existing experiments (`run_sandbox.py`, `run_happy.py`) to get baseline data
2. Implement falsification tests 2–7
3. Implement Phase 5 response law numerics
4. Extend HaPPY code to multi-layer
5. Add JAX acceleration for SRE computation
6. Add data serialization
