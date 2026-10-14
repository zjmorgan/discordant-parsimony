"""Which atoms participate in the correlated disorder?

Candidate units (sets of site labels, typically one split-site cluster
type each) are switched on and off. A unit that does not participate is
uncorrelated with every other position: all its orbit coefficients
beyond the exact within-cluster values are zero. Every combination of
units is refined on training positions and scored on held-out ones, so
models with more parameters gain nothing unless they predict unseen
intensities better.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from discordant_parsimony.constraints import (
    parameterize,
    stack_parameterizations,
)
from discordant_parsimony.refine import r_factors, refine


@dataclass
class ParticipationModel:
    """
    One combination of participating units.

    Attributes
    ----------
    units : tuple of str
        Participating unit names.
    n_params : int
        Refined parameters (correlations, scale, background).
    chi2_train, chi2_test : float
        Reduced chi-squared on training and held-out positions.
    wR_test : float
        Weighted R factor on held-out positions.
    bic : float
        Bayesian information criterion on the training positions.
    refinement : Refinement

    """

    units: tuple
    n_params: int
    chi2_train: float
    chi2_test: float
    wR_test: float
    bic: float
    refinement: object


def participation_scan(
    y,
    sigma,
    A,
    orbits,
    sites,
    units,
    fixed,
    rows,
    train,
    B=None,
    size=None,
):
    """
    Refine every combination of participating units.

    Parameters
    ----------
    y, sigma : ndarray, (m,)
        Observed intensities and uncertainties.
    A : ndarray, (m, n_orbits)
        Design columns.
    orbits : PairOrbits
        Pairs and their orbits.
    sites : Sites
        Participating candidate positions.
    units : dict
        ``{name: [site labels]}`` candidate units.
    fixed : tuple of ndarray
        Coefficients fixed by the average structure.
    rows : ndarray, (r, n_orbits)
        Exact constraints (sum rules, homometric ties).
    train : ndarray of bool, (m,)
        Training positions; the rest are held out.
    B : ndarray, (m, nb), optional
        Background basis.
    size : dict, optional
        Size-effect coefficients, ``{"A": design columns, "orbit": orbit
        of each column, "rows": their constraint rows}``. Every unit
        combination is then refined without and with the size effect
        (labelled ``"size effect"``); its coefficients follow the same
        participation switches as the occupational ones.

    Returns
    -------
    models : list of ParticipationModel
        Sorted by held-out reduced chi-squared.

    """

    label_unit = {}
    for name, labels in units.items():
        for label in labels:
            label_unit[label] = name
    reps = orbits.representatives()
    unit_i = np.array([label_unit.get(sites.label[orbits.i[p]]) for p in reps])
    unit_j = np.array([label_unit.get(sites.label[orbits.j[p]]) for p in reps])
    is_fixed = np.zeros(orbits.n_orbits, bool)
    is_fixed[fixed[0]] = True

    test = ~train
    B = np.zeros((len(y), 0)) if B is None else B
    models = []
    names = list(units)
    sizes = [False, True] if size is not None else [False]
    for k in range(len(names) + 1):
        for chosen in itertools.combinations(names, k):
            on = np.isin(unit_i, chosen) & np.isin(unit_j, chosen)
            off = np.flatnonzero(~on & ~is_fixed)
            zero = np.zeros((len(off), orbits.n_orbits))
            zero[np.arange(len(off)), off] = 1
            p_occ = parameterize(
                orbits.n_orbits, fixed, np.vstack([rows, zero])
            )
            for with_size in sizes:
                if with_size:
                    if not chosen:
                        continue
                    n_vec = size["A"].shape[1]
                    off_vec = np.flatnonzero(~on[size["orbit"]])
                    zero_vec = np.zeros((len(off_vec), n_vec))
                    zero_vec[np.arange(len(off_vec)), off_vec] = 1
                    p_vec = parameterize(
                        n_vec, None, np.vstack([size["rows"], zero_vec])
                    )
                    p = stack_parameterizations(p_occ, p_vec)
                    A_all = np.hstack([A, size["A"]])
                    label = chosen + ("size effect",)
                else:
                    p, A_all, label = p_occ, A, chosen
                r = refine(
                    y[train],
                    sigma[train],
                    A_all[train],
                    B=B[train],
                    parameterization=p,
                )
                model = r.predict(A_all[test], B[test])
                chi2_test = np.sum(((y[test] - model) / sigma[test]) ** 2)
                _, wR = r_factors(y[test], sigma[test], model)
                n = train.sum()
                models.append(
                    ParticipationModel(
                        label,
                        r.n_params,
                        r.reduced_chi2,
                        chi2_test / test.sum(),
                        wR,
                        r.chi2 + r.n_params * np.log(n),
                        r,
                    )
                )
    return sorted(models, key=lambda m: m.chi2_test)
