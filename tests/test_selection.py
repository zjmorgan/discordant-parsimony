import itertools

import numpy as np
import pytest

from discordant_parsimony import Structure
from discordant_parsimony.selection import (
    asymmetric_unit,
    bragg_mask,
    canonical,
    is_allowed,
    laue_operators,
    line_mask,
    satellite_mask,
)


@pytest.fixture
def fcc():
    """Synthetic Fm-3m operator set: 48 rotations x 4 centerings."""

    rotations = []
    for p in itertools.permutations(range(3)):
        for s in itertools.product((-1, 1), repeat=3):
            W = np.zeros((3, 3))
            W[np.arange(3), p] = s
            rotations.append(W)
    centring = [(0, 0, 0), (0, 0.5, 0.5), (0.5, 0, 0.5), (0.5, 0.5, 0)]
    ops = [(W, np.array(t, float)) for W in rotations for t in centring]
    return Structure((10, 10, 10, 90, 90, 90), ops, [])


def test_reflection_conditions(fcc):
    h = np.array([[1, 1, 1], [2, 0, 0], [1, 0, 0], [0.5, 0, 0], [2, 1, 1]])
    assert is_allowed(h, fcc).tolist() == [True, True, False, False, False]


def test_laue_group_size(fcc):
    assert len(laue_operators(fcc)) == 48


def test_bragg_mask(fcc):
    r = 0.06  # inverse Angstrom; 0.1 r.l.u. = 0.0628 here
    h = np.array([[2.0, 0, 0.09], [2.0, 0, 0.11], [1.0, 0, 0]])
    assert bragg_mask(h, fcc, r).tolist() == [True, False, False]


def test_satellites_of_x_point(fcc):
    r = 0.03
    h = np.array([[1, 0, 0], [0, 3, 0], [1, 1, 0], [1, 1, 1], [0.5, 0, 0]])
    mask = satellite_mask(h, fcc, (1, 0, 0), r)
    assert mask.tolist() == [True, True, True, False, False]


def test_lines_through_x_points(fcc):
    r = 0.03
    h = np.array(
        [
            [1, 0.37, 0],  # on (1, t, 0)
            [1, 7.6, 0],  # far along the same line
            [2, 3.3, 1],  # (0, t, 1) + (2, 2, 0)
            [1, 0.37, 0.5],  # half a r.l.u. off the line
            [0.5, 0.5, 0.5],
        ]
    )
    mask = line_mask(h, fcc, (1, 0, 0), (0, 1, 0), r)
    assert mask.tolist() == [True, True, True, False, False]


def test_asymmetric_unit_counts_orbits(fcc):
    n = 3
    g = np.arange(-n, n + 1)
    grid = np.stack(np.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)
    index = tuple((grid + n).T)
    keep = asymmetric_unit(index, (2 * n + 1,) * 3, (n, n, n), fcc)
    W = laue_operators(fcc)
    orbits = {frozenset(map(tuple, (p @ W).tolist())) for p in grid}
    assert keep.sum() == len(orbits)
    reps = grid[keep]
    assert len({frozenset(map(tuple, (p @ W).tolist())) for p in reps}) == len(
        orbits
    )


def test_canonical_is_shared_by_equivalents(fcc):
    rng = np.random.default_rng(0)
    h = np.round(rng.normal(size=(20, 3)) * 4, 2)
    rep = canonical(h, fcc)
    for W in laue_operators(fcc)[::7]:
        assert np.allclose(canonical(h @ W, fcc), rep)
