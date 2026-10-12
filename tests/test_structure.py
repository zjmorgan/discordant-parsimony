import numpy as np
import pytest

from discordant_parsimony.structure import lattice_vectors, parse_symop

from conftest import requires_tb5ru6sn18


def test_parse_symop():
    W, t = parse_symop("-x+1/2, y-x, 1/4+z")
    assert np.allclose(W, [[-1, 0, 0], [-1, 1, 0], [0, 0, 1]])
    assert np.allclose(t, [0.5, 0, 0.25])
    with pytest.raises(ValueError):
        parse_symop("x, y")


def test_lattice_vectors_reciprocal_convention():
    A = lattice_vectors(4.0, 5.0, 6.0, 80, 95, 110)
    assert np.allclose(np.linalg.norm(A, axis=0), [4, 5, 6])
    B = np.linalg.inv(A).T
    assert np.allclose(A.T @ B, np.eye(3))
    rng = np.random.default_rng(0)
    h, x = rng.normal(size=3), rng.normal(size=3)
    assert np.isclose((2 * np.pi * B @ h) @ (A @ x), 2 * np.pi * h @ x)


def test_expansion_and_adp_rotation(tetragonal):
    sites = tetragonal.sites()
    assert list(sites.label).count("M1") == 1
    assert list(sites.label).count("A1") == 4
    assert sites.partial() == ["A1", "A2"]
    a1 = sites.select(["A1"])
    first = np.flatnonzero(np.all(np.isclose(a1.x, [0.1, 0.2, 0.3]), axis=1))
    assert len(first) == 1
    U = a1.U[first[0]]
    assert np.allclose(np.diag(U), [0.01, 0.02, 0.015])
    assert np.isclose(U[0, 1], 0.003)
    # 4-fold rotation (-y, x, z) maps (0.1, 0.2) to (0.8, 0.1)
    k = np.flatnonzero(np.all(np.isclose(a1.x, [0.8, 0.1, 0.3]), axis=1))[0]
    assert np.allclose(np.diag(a1.U[k]), [0.02, 0.01, 0.015])
    assert np.isclose(a1.U[k][0, 1], -0.003)
    a2 = sites.select(["A2"])
    assert np.allclose(a2.U, 0.01 * np.eye(3))
    with pytest.raises(KeyError):
        sites.select(["X9"])


@requires_tb5ru6sn18
def test_tb5ru6sn18_sites(tb5ru6sn18):
    sites = tb5ru6sn18.sites()
    counts = {l: list(sites.label).count(l) for l in set(sites.label)}
    assert counts == {
        "Tb1": 4,
        "Tb2": 32,
        "Ru1": 24,
        "Sn1": 32,
        "Sn2": 48,
        "Sn3": 48,
    }
    per_cell = {}
    for species, occ in zip(sites.species, sites.occupancy):
        per_cell[species] = per_cell.get(species, 0) + occ
    # Sn2 occupancy is given to four decimals (0.1667) in the CIF
    composition = [per_cell[s] for s in ("Tb", "Ru", "Sn")]
    assert np.allclose(composition, [20, 24, 72], atol=0.01)
    assert sites.partial() == ["Tb2", "Sn1", "Sn2"]
