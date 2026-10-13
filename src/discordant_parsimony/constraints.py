r"""Exact linear constraints on orbit coefficients and their solution.

Two kinds of relation are imposed, both independent of any model of the
local order:

* sum rules: a split-site cluster without vacancies always holds exactly
  one atom, so the fluctuations of its members sum to zero and their
  covariances with any position sum to zero,
  :math:`\sum_a C(a \,@\, \mathbf{o}_a,\, j \,@\, \mathbf{R}) = 0`;
* homometric ties: orbits whose pair vectors, scattering-length products
  and displacement tensors coincide give identical intensities, so only
  their sum is measurable; they are tied equal (each then carries the
  average of the group).

Together with coefficients fixed by the average structure they define
:math:`\mathbf{c} = \mathbf{c}_0 + N \boldsymbol{\theta}` with free
parameters :math:`\boldsymbol{\theta}`.
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import null_space


def _pair_lookup(orbits):
    return {
        (i, j, tuple(R)): o
        for i, j, R, o in zip(orbits.i, orbits.j, orbits.R, orbits.orbit)
    }


def sum_rule_rows(orbits, clusters, vacancy_tol=1e-3):
    """
    Sum-rule constraints of clusters without vacancies.

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    clusters : list of Cluster
        Split-site clusters covering all positions.
    vacancy_tol : float, optional
        Clusters with ``vacancy`` above this get no rule.

    Returns
    -------
    rows : ndarray, (r, n_orbits)
        Each row ``M`` states ``M @ c = 0``. Rows that would involve
        pairs beyond the cutoff are omitted.

    """

    lookup = _pair_lookup(orbits)
    rows = set()
    for cluster in clusters:
        if cluster.vacancy > vacancy_tol or len(cluster) < 2:
            continue
        a0, o0 = cluster.members[0], cluster.offsets[0]
        for j, R0 in zip(orbits.j[orbits.i == a0], orbits.R[orbits.i == a0]):
            R = R0 + o0
            terms = {}
            for a, oa in zip(cluster.members, cluster.offsets):
                o = lookup.get((a, j, tuple(R - oa)))
                if o is None:
                    break
                terms[o] = terms.get(o, 0) + 1
            else:
                rows.add(tuple(sorted(terms.items())))
    M = np.zeros((len(rows), orbits.n_orbits))
    for r, row in enumerate(rows):
        for o, v in row:
            M[r, o] = v
    return M


def homometric_groups(orbits, sites, decimals=6):
    """
    Groups of orbits that give identical intensities.

    Parameters
    ----------
    orbits : PairOrbits
        Pairs and their orbits.
    sites : Sites
        Participating positions (scattering lengths and ADPs).
    decimals : int, optional
        Rounding used to compare pair vectors and tensors.

    Returns
    -------
    group : ndarray of int, (n_orbits,)
        Group label of each orbit; singletons label themselves.

    """

    b = sites.b
    U = sites.U[orbits.i] + sites.U[orbits.j]
    iu = np.triu_indices(3)
    rows = np.hstack(
        [
            orbits.d,
            (b[orbits.i] * b[orbits.j])[:, None],
            U[:, iu[0], iu[1]],
        ]
    )
    rows = np.round(rows, decimals)
    signature = {}
    group = np.arange(orbits.n_orbits)
    for o in range(orbits.n_orbits):
        r = rows[orbits.orbit == o]
        key = r[np.lexsort(r.T[::-1])].tobytes()
        group[o] = signature.setdefault(key, o)
    return group


def tie_rows(group, exclude=()):
    """Rows ``c_o - c_g = 0`` tying each orbit to its group label."""

    exclude = set(np.atleast_1d(exclude).tolist())
    pairs = [
        (o, g)
        for o, g in enumerate(group)
        if o != g and o not in exclude and g not in exclude
    ]
    M = np.zeros((len(pairs), len(group)))
    for r, (o, g) in enumerate(pairs):
        M[r, o], M[r, g] = 1, -1
    return M


@dataclass
class Parameterization:
    r"""
    Coefficients :math:`\mathbf{c} = \mathbf{c}_0 + N \boldsymbol{\theta}`
    satisfying all constraints.

    Attributes
    ----------
    c0 : ndarray, (n_orbits,)
        Fixed values plus the particular solution of the constraints.
    N : ndarray, (n_orbits, k)
        Orthonormal basis of free directions (zero on fixed orbits).
    residual : float
        Largest constraint violation of ``c0`` (from rounding of the
        average-structure occupancies).

    """

    c0: np.ndarray
    N: np.ndarray
    residual: float

    @property
    def n_free(self):
        return self.N.shape[1]

    def coefficients(self, theta):
        return self.c0 + self.N @ theta


def parameterize(n_orbits, fixed=None, rows=None, tol=1e-3):
    """
    Solve fixed values and linear constraints for free directions.

    Parameters
    ----------
    n_orbits : int
        Number of orbit coefficients.
    fixed : tuple of ndarray, optional
        ``(index, value)`` of fixed coefficients.
    rows : ndarray, (r, n_orbits), optional
        Constraint rows ``M @ c = 0``.
    tol : float, optional
        Largest tolerated violation of the constraints by the fixed
        values.

    Returns
    -------
    parameterization : Parameterization

    """

    free = np.ones(n_orbits, bool)
    c0 = np.zeros(n_orbits)
    if fixed is not None and len(fixed[0]):
        free[fixed[0]] = False
        c0[fixed[0]] = fixed[1]
    N = np.zeros((n_orbits, free.sum()))
    N[free] = np.eye(free.sum())
    residual = 0.0
    if rows is not None and len(rows):
        Mf, Mx = rows[:, free], rows[:, ~free]
        rhs = -Mx @ c0[~free]
        particular = np.linalg.lstsq(Mf, rhs, rcond=None)[0]
        residual = float(np.max(np.abs(Mf @ particular - rhs), initial=0))
        if residual > tol:
            raise ValueError(f"Fixed values violate constraints ({residual})")
        c0[free] = particular
        basis = null_space(Mf)
        N = np.zeros((n_orbits, basis.shape[1]))
        N[free] = basis
    return Parameterization(c0, N, residual)
