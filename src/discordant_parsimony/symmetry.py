"""Symmetry-independent pair correlations.

A pair ``(i, j, R)`` joins position ``i`` in the origin cell to position
``j`` in cell ``R``. A space-group operator ``(W, t)`` maps it to
``(i', j', R')`` with ``W x_i + t = x_i' + L_i``, ``W x_j + t = x_j' +
L_j`` and ``R' = W R + L_j - L_i``; pair reversal maps it to
``(j, i, -R)``. Correlations with the full symmetry of the average
structure are equal on each orbit, so one coefficient per orbit
describes them. Pairs are selected with a distance cutoff, which is
invariant under the group.
"""

from dataclasses import dataclass

import numpy as np


def position_permutations(structure, sites, tol=1e-4):
    """
    Action of each operator on the positions of one cell.

    Parameters
    ----------
    structure : Structure
        Supplies the symmetry operators.
    sites : Sites
        Positions closed under the operators.
    tol : float, optional
        Fractional matching tolerance.

    Returns
    -------
    perm : ndarray of int, (g, n)
        ``perm[g, i]`` is the image position of ``i``.
    shift : ndarray of int, (g, n, 3)
        Lattice vector with ``W x_i + t = x_perm + shift``.

    """

    n = len(sites)
    perm = np.empty((len(structure.operators), n), int)
    shift = np.empty((len(structure.operators), n, 3), int)
    for g, (W, t) in enumerate(structure.operators):
        image = sites.x @ W.T + t
        diff = image[:, None] - sites.x[None]
        frac = (diff + 0.5) % 1 - 0.5
        match = np.all(np.abs(frac) < tol, axis=2)
        match &= sites.label[None] == sites.label[:, None]
        if not np.all(match.sum(axis=1) == 1):
            raise ValueError("Positions are not closed under the operators")
        perm[g] = np.argmax(match, axis=1)
        shift[g] = np.rint(image - sites.x[perm[g]]).astype(int)
    return perm, shift


@dataclass
class PairOrbits:
    """
    Pairs within a cutoff, grouped into symmetry orbits.

    Attributes
    ----------
    i, j : ndarray of int, (m,)
        Position indices.
    R : ndarray of int, (m, 3)
        Cell translation of ``j``.
    d : ndarray, (m, 3)
        Pair vectors ``R + x_j - x_i`` (fractional).
    orbit : ndarray of int, (m,)
        Orbit index of each pair, ``0 .. n_orbits - 1``, ordered by
        pair distance.
    distance : ndarray, (n_orbits,)
        Pair distance of each orbit in Angstrom.

    """

    i: np.ndarray
    j: np.ndarray
    R: np.ndarray
    d: np.ndarray
    orbit: np.ndarray
    distance: np.ndarray

    def __len__(self):
        return len(self.i)

    @property
    def n_orbits(self):
        return len(self.distance)

    def representatives(self):
        """Index of the first pair of each orbit."""

        _, first = np.unique(self.orbit, return_index=True)
        return first


def pair_orbits(structure, sites, max_distance, tol=1e-6):
    """
    All pairs up to a distance, grouped into symmetry orbits.

    Parameters
    ----------
    structure : Structure
        Supplies lattice and operators.
    sites : Sites
        Participating positions, closed under the operators.
    max_distance : float
        Pair distance cutoff in Angstrom (self pairs included).
    tol : float, optional
        Distance tolerance at the cutoff.

    Returns
    -------
    orbits : PairOrbits

    """

    n = np.ceil(max_distance * np.linalg.norm(structure.B, axis=0)) + 1
    ranges = [np.arange(-k, k + 1) for k in n.astype(int)]
    T = np.stack(np.meshgrid(*ranges, indexing="ij"), -1).reshape(-1, 3)
    dx = sites.x[None, :, None] - sites.x[:, None, None] + T
    i, j, k = np.nonzero(structure.distance(dx) <= max_distance + tol)
    R = T[k]

    span = 2 * int(np.abs(R).max()) + 2 * int(np.max(n)) + 3
    offset = span // 2

    def encode(i, j, R):
        Rc = R + offset
        if np.any((Rc < 0) | (Rc >= span)):
            raise ValueError("Pair image outside the encoding range")
        cell = (Rc[:, 0] * span + Rc[:, 1]) * span + Rc[:, 2]
        return (i * len(sites) + j) * span**3 + cell

    key = encode(i, j, R)
    order = np.argsort(key)
    sorted_key = key[order]

    perm, shift = position_permutations(structure, sites)
    label = key.copy()
    for g, (W, _) in enumerate(structure.operators):
        ip, jp = perm[g, i], perm[g, j]
        Rp = R @ np.rint(W).astype(int).T + shift[g, j] - shift[g, i]
        for a, b, Rq in ((ip, jp, Rp), (jp, ip, -Rp)):
            image = encode(a, b, Rq)
            pos = np.searchsorted(sorted_key, image)
            found = (pos < len(key)) & (
                sorted_key[np.minimum(pos, len(key) - 1)] == image
            )
            if not np.all(found):
                raise ValueError("Pair set not closed under symmetry")
            label = np.minimum(label, image)

    d = R + sites.x[j] - sites.x[i]
    dist = structure.distance(d)
    _, first, inverse = np.unique(
        label, return_index=True, return_inverse=True
    )
    rank = np.lexsort((first, np.round(dist[first], 6)))
    order_of = np.empty_like(rank)
    order_of[rank] = np.arange(len(rank))
    orbit = order_of[inverse.ravel()]
    return PairOrbits(i, j, R, d, orbit, dist[first][rank])


def orbit_average(orbits, R, C):
    """
    Project a covariance onto orbit coefficients (symmetrize).

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    R : ndarray of int, (k, 3)
        Translations of the covariance array.
    C : ndarray, (k, n, n)
        Covariance ``C[k, i, j] = <dn_i(0) dn_j(R_k)>``.

    Returns
    -------
    c : ndarray, (n_orbits,)
        Mean of ``C`` over the pairs of each orbit.

    """

    index = {tuple(r): k for k, r in enumerate(R)}
    try:
        k = np.array([index[tuple(r)] for r in orbits.R])
    except KeyError as error:
        raise ValueError(
            "Covariance lacks a translation of the pairs"
        ) from error
    values = C[k, orbits.i, orbits.j]
    total = np.bincount(
        orbits.orbit, weights=values, minlength=orbits.n_orbits
    )
    count = np.bincount(orbits.orbit, minlength=orbits.n_orbits)
    return total / count


def orbit_pair_terms(orbits, sites, c, thermal=True, b=None):
    """
    Pair terms of one covariance coefficient per orbit.

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    sites : Sites
        Participating positions.
    c : array_like, (n_orbits,)
        Occupation covariance of each orbit.
    thermal : bool, optional
        Smear by ``U_i + U_j``.
    b : array_like, (n,), optional
        Scattering lengths in fm. Defaults to neutron values.

    Returns
    -------
    terms : PairTerms

    """

    from discordant_parsimony.pairs import PairTerms

    b = sites.b if b is None else np.asarray(b)
    w = b[orbits.i] * b[orbits.j] * np.asarray(c)[orbits.orbit]
    U = sites.U[orbits.i] + sites.U[orbits.j] if thermal else None
    return PairTerms(orbits.d, w, U)
