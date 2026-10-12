"""Occupation covariances ``C_ij(R) = <dn_i(0) dn_j(R)>``.

Covariances may come from an analytic model or be estimated from
occupation configurations of a periodic lattice of cells. Only the
participating positions are represented; this is not an atomistic
supercell of the full structure.
"""

import numpy as np


def translations(max_offset):
    """
    Cell translations with ``|R_a| <= max_offset`` along each axis.

    Parameters
    ----------
    max_offset : int or sequence of int
        Cell-index cutoff per axis.

    Returns
    -------
    R : ndarray of int, (m, 3)

    """

    n = np.broadcast_to(max_offset, 3)
    ranges = [np.arange(-k, k + 1) for k in n]
    return np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, 3)


def covariance_from_configurations(occupation, max_offset, mean=None):
    """
    Estimate occupation covariances from periodic configurations.

    Parameters
    ----------
    occupation : ndarray, (m, L1, L2, L3, n)
        Occupation (0 or 1) of each position in each cell for ``m``
        independent configurations.
    max_offset : int or sequence of int
        Cell-index cutoff; ``2 * max_offset + 1`` must not exceed the
        lattice size so that no translation is counted twice.
    mean : array_like, (n,), optional
        Average occupation. Estimated from the configurations by
        default.

    Returns
    -------
    R : ndarray of int, (k, 3)
        Cell translations.
    C : ndarray, (k, n, n)
        ``C[k, i, j] = <dn_i(0) dn_j(R_k)>`` averaged over cells and
        configurations.

    """

    occupation = np.asarray(occupation, dtype=float)
    shape = np.array(occupation.shape[1:4])
    R = translations(max_offset)
    if np.any(2 * np.abs(R).max(axis=0) + 1 > shape):
        raise ValueError("max_offset too large for the lattice size")

    if mean is None:
        mean = occupation.mean(axis=(0, 1, 2, 3))
    delta = occupation - mean
    F = np.fft.fftn(delta, axes=(1, 2, 3))
    index = tuple((R % shape).T)

    n = occupation.shape[-1]
    C = np.empty((len(R), n, n))
    for i in range(n):
        corr = np.fft.ifftn(F[..., i, None].conj() * F, axes=(1, 2, 3))
        C[:, i, :] = corr.real.mean(axis=0)[index] / np.prod(shape)
    return R, C


def independent_cluster_covariance(sites, clusters):
    """
    Covariance of independent split-site clusters (no local order).

    Within a cluster at most one member is occupied, so
    ``C_ab = x_a delta_ab - x_a x_b``; different clusters are
    uncorrelated.

    Parameters
    ----------
    sites : Sites
        Participating positions.
    clusters : list of Cluster
        Split-site clusters covering all positions.

    Returns
    -------
    R : ndarray of int, (k, 3)
        Cell translations present.
    C : ndarray, (k, n, n)
        Occupation covariance.

    """

    entries = {}
    for cluster in clusters:
        x = sites.occupancy[cluster.members]
        cov = np.diag(x) - np.outer(x, x)
        for a, (i, oi) in enumerate(zip(cluster.members, cluster.offsets)):
            for c, (j, oj) in enumerate(zip(cluster.members, cluster.offsets)):
                entries.setdefault(tuple(oj - oi), []).append(
                    (i, j, cov[a, c])
                )

    R = np.array(sorted(entries))
    n = len(sites)
    C = np.zeros((len(R), n, n))
    for k, key in enumerate(map(tuple, R)):
        for i, j, value in entries[key]:
            C[k, i, j] += value
    return R, C
