"""Tests for the analytical response-law module.

Verifies closed-form and spectral results against the numerical FWHT SRE.
"""

import numpy as np
import pytest

from mqhg.analytical import (
    sre_single_triplet,
    sre_disjoint_triplets,
    sre_ipr_decomposition,
    single_triplet_polynomial,
    local_response_exponent,
)
from mqhg.measures.magic import stabilizer_renyi_entropy
from mqhg.states.hypergraph import HypergraphState, Hypergraph


THETAS = [0.0, 0.3, 0.5, 1.0, np.pi / 2, 2.0, 2.4, 2.8, np.pi]


def _single_triplet_state(theta):
    hg = Hypergraph(3)
    hg.add_edge((0, 1, 2), theta)
    return HypergraphState(hg).prepare()


@pytest.mark.parametrize("theta", THETAS)
def test_single_triplet_closed_form(theta):
    """Closed form matches numerical SRE for CCZ(theta)|+++>."""
    state = _single_triplet_state(theta)
    numeric = stabilizer_renyi_entropy(state)
    closed = sre_single_triplet(theta)
    assert np.isclose(numeric, closed, atol=1e-9), f"{numeric} != {closed}"


def test_single_triplet_zero_at_origin():
    """Stabilizer point theta=0 gives zero magic."""
    assert np.isclose(sre_single_triplet(0.0), 0.0, atol=1e-12)


def test_single_triplet_polynomial_endpoints():
    """Polynomial P(cos t) hits known integer values."""
    assert np.isclose(single_triplet_polynomial(0.0), 256.0)  # P(1)
    assert np.isclose(single_triplet_polynomial(np.pi), 88.0)  # P(-1)
    assert np.isclose(single_triplet_polynomial(np.pi / 2), 109.0)  # P(0)


@pytest.mark.parametrize("theta", [0.3, 1.0, np.pi / 2, 2.4])
def test_disjoint_additivity(theta):
    """Disjoint triplets give additive SRE."""
    hg = Hypergraph(9)
    hg.add_edge((0, 1, 2), theta)
    hg.add_edge((3, 4, 5), theta)
    hg.add_edge((6, 7, 8), theta)
    numeric = stabilizer_renyi_entropy(HypergraphState(hg).prepare())
    analytical = sre_disjoint_triplets(theta, 3)
    assert np.isclose(numeric, analytical, atol=1e-9)


@pytest.mark.parametrize("theta", [0.5, np.pi / 2, np.pi])
def test_clifford_invariance_ring(theta):
    """Adding ring CZ edges does not change SRE (Clifford invariance)."""
    n = 6
    hg_ccz = Hypergraph(n)
    for i in range(0, n, 2):
        hg_ccz.add_edge((i, (i + 1) % n, (i + 2) % n), theta)
    sre_ccz = stabilizer_renyi_entropy(HypergraphState(hg_ccz).prepare())

    hg_ring = Hypergraph(n)
    for i in range(n):
        hg_ring.add_edge((i, (i + 1) % n), np.pi)
    for i in range(0, n, 2):
        hg_ring.add_edge((i, (i + 1) % n, (i + 2) % n), theta)
    sre_ring = stabilizer_renyi_entropy(HypergraphState(hg_ring).prepare())

    assert np.isclose(sre_ccz, sre_ring, atol=1e-9)


@pytest.mark.parametrize("theta", [0.5, 1.5, 2.4])
def test_ipr_decomposition_matches_direct(theta):
    """IPR decomposition reproduces the direct FWHT SRE."""
    n = 6
    hg = Hypergraph(n)
    for i in range(n):
        hg.add_edge((i, (i + 1) % n), np.pi)
    for i in range(0, n, 2):
        hg.add_edge((i, (i + 1) % n, (i + 2) % n), theta)
    state = HypergraphState(hg).prepare()

    m2_ipr, ipr = sre_ipr_decomposition(state)
    m2_direct = stabilizer_renyi_entropy(state)
    assert np.isclose(m2_ipr, m2_direct, atol=1e-9)
    assert len(ipr) == 2**n
    assert np.all(ipr <= 1.0 + 1e-9)


def test_ipr_stabilizer_all_ones():
    """Pure stabilizer (ring graph) has IPR=1 for every flip mask."""
    n = 5
    hg = Hypergraph(n)
    for i in range(n):
        hg.add_edge((i, (i + 1) % n), np.pi)
    state = HypergraphState(hg).prepare()
    m2, ipr = sre_ipr_decomposition(state)
    assert np.isclose(m2, 0.0, atol=1e-9)
    assert np.allclose(ipr, 1.0, atol=1e-9)


def test_local_exponent_linear_at_small_theta():
    """Local response exponent approaches 1 (linear) for small magic."""
    n = 6
    thetas = np.linspace(0.05, 0.6, 20)
    from mqhg.measures.entanglement import mutual_information_matrix

    magic, backreaction = [], []
    for th in thetas:
        hg = Hypergraph(n)
        for i in range(n):
            hg.add_edge((i, (i + 1) % n), np.pi)
        for i in range(0, n, 2):
            hg.add_edge((i, (i + 1) % n, (i + 2) % n), th)
        st = HypergraphState(hg).prepare()
        magic.append(stabilizer_renyi_entropy(st))
        backreaction.append(np.linalg.norm(mutual_information_matrix(st), "fro"))

    alpha = local_response_exponent(np.array(magic), np.array(backreaction))
    # In the weak-magic regime the response is linear (alpha ~ 1).
    assert np.nanmean(alpha) == pytest.approx(1.0, abs=0.1)
