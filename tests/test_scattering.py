import numpy as np
import pytest

from discordant_parsimony import (
    PairTerms,
    evaluate,
    pair_terms_from_covariance,
    thermal_laue_terms,
)
from discordant_parsimony.correlations import (
    covariance_from_configurations,
    independent_cluster_covariance,
)
from discordant_parsimony.rules import find_clusters


def correlated_occupations(rng, m, L, n):
    """Binary fields with short-range correlations within and between
    positions: thresholded, locally averaged, mixed Gaussian noise."""

    z = rng.normal(size=(m, L, L, L, n))
    z = z + 0.6 * np.roll(z, 1, axis=1) + 0.4 * np.roll(z, -1, axis=3)
    z = z @ (np.eye(n) + 0.5 * rng.normal(size=(n, n)))
    return z > 0.3


def brute_force(structure, sites, occupation, h):
    """(1/N) <|F - <F>|^2> of explicit periodic configurations with
    Debye-Waller factors on every atom."""

    m, L = occupation.shape[0], occupation.shape[1]
    delta = occupation - occupation.mean(axis=(0, 1, 2, 3))
    R = np.stack(np.meshgrid(*[np.arange(L)] * 3, indexing="ij"), -1)
    Q = structure.Q(h)
    I = np.zeros(len(h))
    for q, (hq, Qq) in enumerate(zip(h, Q)):
        dw = np.exp(-0.5 * np.einsum("a,pab,b->p", Qq, sites.U, Qq))
        phase = np.exp(2j * np.pi * ((R[..., None, :] + sites.x) @ hq))
        F = np.einsum("mabcp,abcp,p->m", delta, phase, sites.b * dw)
        I[q] = np.mean(np.abs(F) ** 2) / L**3
    return I


@pytest.fixture
def setup(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    rng = np.random.default_rng(7)
    L = 5
    occupation = correlated_occupations(rng, 4, L, len(sites))
    h = rng.integers(-12, 13, size=(25, 3)) / L
    return tetragonal, sites, occupation, h


def test_pair_sum_equals_supercell(setup):
    structure, sites, occupation, h = setup
    R, C = covariance_from_configurations(occupation, 2)
    terms = pair_terms_from_covariance(sites, R, C)
    I = evaluate(h, terms, structure, max_elements=5000, check=True)
    I0 = brute_force(structure, sites, occupation, h)
    assert np.allclose(I, I0, rtol=1e-10, atol=1e-10 * I0.max())


def test_merge_preserves_intensity(setup):
    structure, sites, occupation, h = setup
    R, C = covariance_from_configurations(occupation, 2)
    terms = pair_terms_from_covariance(sites, R, C)
    merged = terms.merge()
    assert len(merged) < 0.6 * len(terms)
    assert np.allclose(
        evaluate(h, merged, structure), evaluate(h, terms, structure)
    )


def test_bin_average_is_exact_for_point_terms(setup):
    structure, sites, occupation, _ = setup
    R, C = covariance_from_configurations(occupation, 1)
    terms = pair_terms_from_covariance(sites, R, C, thermal=False)
    widths = np.array([0.25, 0.2, 0.3])
    centers = np.array([[1.0, -0.5, 2.25], [0.3, 0.7, -1.1]])
    k = 12
    u = (np.arange(k) + 0.5) / k - 0.5
    sub = np.stack(np.meshgrid(u, u, u, indexing="ij"), -1).reshape(-1, 3)
    for c in centers:
        numeric = evaluate(c + sub * widths, terms, structure).mean()
        exact = evaluate(c, terms.bin_average(widths), structure)
        assert np.isclose(numeric, exact, rtol=2e-3)


def test_unclosed_terms_rejected(tetragonal):
    terms = PairTerms(np.array([[0.1, 0.2, 0.3]]), np.array([1.0]))
    with pytest.raises(ValueError):
        evaluate(np.array([[0.4, 0.1, 0.2]]), terms, tetragonal, check=True)


def test_zero_covariance_gives_zero(setup):
    structure, sites, _, h = setup
    R = np.zeros((1, 3), int)
    C = np.zeros((1, len(sites), len(sites)))
    terms = pair_terms_from_covariance(sites, R, C)
    assert np.allclose(evaluate(h, terms, structure), 0)


def test_independent_atom_self_term(tetragonal):
    """Uncorrelated split sites with Gaussian motion: the exact result
    sum_i b_i^2 (x_i - x_i^2 e^{-2W_i}) minus within-cluster cross
    terms 2 x_a x_b b_a b_b e^{-W_a-W_b} cos(Q.d_ab)."""

    sites = tetragonal.sites().select(["A1", "A2"])
    clusters = find_clusters(tetragonal, sites, 1.0)
    R, C = independent_cluster_covariance(sites, clusters)
    terms = PairTerms.concatenate(
        [pair_terms_from_covariance(sites, R, C), thermal_laue_terms(sites)]
    )
    h = np.array([[0.3, 1.2, -2.1], [3.0, 0.0, 1.0], [-1.7, 2.2, 0.4]])
    Q = tetragonal.Q(h)
    W = 0.5 * np.einsum("qa,pab,qb->qp", Q, sites.U, Q)
    x, b = sites.occupancy, sites.b
    expected = (b**2 * (x - x**2 * np.exp(-2 * W))).sum(axis=1)
    for c in clusters:
        i, j = c.members
        d = sites.x[j] + c.offsets[1] - sites.x[i] - c.offsets[0]
        expected -= (
            2
            * x[i]
            * x[j]
            * b[i]
            * b[j]
            * np.exp(-W[:, i] - W[:, j])
            * np.cos(2 * np.pi * h @ d)
        )
    assert np.allclose(evaluate(h, terms, tetragonal), expected)
