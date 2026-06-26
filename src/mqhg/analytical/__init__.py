"""Analytical results for the magic-geometry response law.

Closed-form and spectral-decomposition results for stabilizer Renyi entropy
(SRE) and the magic-geometry response of hypergraph states, derived in
docs/response_law_derivation.md and verified against the numerical FWHT
pipeline in mqhg.measures.magic.
"""

from .response_law import (
    sre_ipr_decomposition,
    sre_single_triplet,
    sre_disjoint_triplets,
    single_triplet_polynomial,
    local_response_exponent,
)

__all__ = [
    "sre_ipr_decomposition",
    "sre_single_triplet",
    "sre_disjoint_triplets",
    "single_triplet_polynomial",
    "local_response_exponent",
]
