import numpy as np
import pytest

from discordant_parsimony.rules import (
    find_clusters,
    find_contacts,
    forbidden_contacts,
)

from conftest import requires_tb5ru6sn18


def test_contacts_are_symmetric(tetragonal):
    sites = tetragonal.sites()
    c = find_contacts(tetragonal, sites, 3.0)
    forward = {(i, j, tuple(R)) for i, j, R in zip(c.i, c.j, c.R)}
    assert all((j, i, tuple(-np.array(R))) in forward for i, j, R in forward)
    dx = sites.x[c.j] + c.R - sites.x[c.i]
    assert np.allclose(tetragonal.distance(dx), c.distance)


def test_split_pair_clusters(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    # A1 (z=0.3) and A2 (z=0.5) are 0.8 A apart: four split pairs
    clusters = find_clusters(tetragonal, sites, 1.0)
    assert sorted(len(c) for c in clusters) == [2, 2, 2, 2]
    assert all(np.isclose(c.vacancy, 0.25) for c in clusters)
    # the split partners are excluded from the forbidden contacts
    fc = forbidden_contacts(tetragonal, sites, clusters, 1.0)
    assert len(fc) == 0


def test_overfull_cluster_raises(tetragonal):
    sites = tetragonal.sites().select(["A1", "A2"])
    sites.occupancy[:] = 0.75
    with pytest.raises(ValueError):
        find_clusters(tetragonal, sites, 1.0)


@requires_tb5ru6sn18
def test_tb5ru6sn18_rules(tb5ru6sn18):
    sites = tb5ru6sn18.sites()
    active = sites.select(sites.partial())
    clusters = find_clusters(tb5ru6sn18, active, 1.2)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [2] * 32 + [6] * 8
    assert all(c.vacancy == 0 for c in clusters)
    limits = {("Sn", "Sn"): 2.6, ("Sn", "Tb"): 2.8, ("Tb", "Tb"): 3.0}
    fc = forbidden_contacts(tb5ru6sn18, active, clusters, limits)
    assert fc.summary(active) == [
        ("Sn1", "Sn1", 2.44, 3.0),
        ("Sn2", "Tb2", 2.511, 2.0),
        ("Tb2", "Sn2", 2.511, 3.0),
    ]
