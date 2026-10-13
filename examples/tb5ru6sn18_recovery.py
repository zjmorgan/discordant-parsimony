"""Synthetic recovery of C_ij(R) from incomplete Tb5Ru6Sn18 data.

Known orbit coefficients are taken from the 2:2 pseudospin states and
used to generate noisy intensities on the measured grid. The
coefficients are then refined from deliberately incomplete subsets and
compared with the truth, on held-out positions, and as a 3D-delta-PDF.
"""

from pathlib import Path

import numpy as np

from discordant_parsimony.correlations import (
    covariance_from_configurations,
    fixed_cluster_orbits,
)
from discordant_parsimony.io import reduce_volume
from discordant_parsimony.realspace import delta_pdf_slab
from discordant_parsimony.refine import background_basis, r_factors, refine
from discordant_parsimony.scattering import design_matrix
from discordant_parsimony.selection import (
    bragg_mask,
    canonical,
    line_mask,
)
from discordant_parsimony.symmetry import (
    orbit_average,
    orbit_pair_terms,
    pair_orbits,
)

from tb5ru6sn18_ice import (
    MAX_OFFSET,
    active,
    clusters,
    configurations,
    root,
    structure,
)

CUTOFF = 14.0  # Angstrom
H_MAX = 12  # r.l.u. half-width of the region considered
BRAGG_RADIUS = 0.14  # inverse Angstrom
NOISE = 0.03  # relative noise level


def main():
    rng = np.random.default_rng(1)
    orbits = pair_orbits(structure, active, CUTOFF)
    R, C = covariance_from_configurations(configurations(), MAX_OFFSET)
    c_true = orbit_average(orbits, R, C)
    fixed = fixed_cluster_orbits(orbits, active, clusters)
    c_true[fixed[0]] = fixed[1]
    print(f"{orbits.n_orbits} orbits within {CUTOFF} A, {len(fixed[0])} fixed")

    volume = reduce_volume(
        *[
            next(root.glob(f"*m-3m_{tag}.nxs"))
            for tag in ("data", "norm", "bkg_data", "bkg_norm")
        ]
    )
    index = np.nonzero(np.isfinite(volume.signal) & (volume.sigma > 0))
    h = volume.hkl(index)
    inside = np.all(np.abs(h) <= H_MAX, axis=1)
    h = h[inside]
    h = h[~bragg_mask(h, structure, BRAGG_RADIUS)]

    # one row per Laue orbit (the data are m-3m symmetrized)
    rep, first, inverse = np.unique(
        canonical(h, structure), axis=0, return_index=True, return_inverse=True
    )
    print(f"{len(rep)} independent positions with |h_a| <= {H_MAX}")
    A = design_matrix(rep, orbits, active, structure, widths=volume.widths)
    Q = np.linalg.norm(structure.Q(rep), axis=1)
    B = background_basis(Q, degree=1)

    y0 = A @ c_true + B @ np.array([0.02, 0.01]) * np.abs(A @ c_true).max()
    sigma = NOISE * (np.median(np.abs(y0)) + np.abs(y0))
    y = y0 + sigma * rng.normal(size=len(y0))

    inverse = inverse.ravel()
    on_line = line_mask(rep, structure, (1, 0, 0), (0, 1, 0), 0.08)
    subsets = {
        "complete (|h|<=12)": np.ones(len(rep), bool),
        "random 10%": rng.random(len(rep)) < 0.10,
        "random 2%": rng.random(len(rep)) < 0.02,
        "(h,k,1) plane only": np.isin(
            np.arange(len(rep)), np.unique(inverse[np.isclose(h[:, 2], 1)])
        ),
        "streak lines + 2% background": on_line
        | (rng.random(len(rep)) < 0.02),
        "streak lines only": on_line,
    }

    free = np.setdiff1d(np.arange(orbits.n_orbits), fixed[0])
    scale = np.sqrt(np.mean(c_true[free] ** 2))
    results = {}
    print(
        f"{'subset':32s} {'N':>6s} {'rank':>5s} {'coef err':>9s}"
        f" {'|z|<3':>6s} {'wR held-out':>11s}"
    )
    for name, mask in subsets.items():
        r = refine(y[mask], sigma[mask], A[mask], fixed=fixed, B=B[mask])
        err = np.sqrt(np.mean((r.c[free] - c_true[free]) ** 2)) / scale
        z = (r.c[free] - c_true[free]) / np.where(
            r.esd[free] > 0, r.esd[free], np.inf
        )
        held = ~mask if (~mask).any() else mask
        _, wR = r_factors(y[held], sigma[held], r.predict(A[held], B[held]))
        print(
            f"{name:32s} {mask.sum():6d} {r.rank:3d}/{r.n_params:<3d}"
            f" {err:9.3f} {np.mean(np.abs(z) < 3):6.2f} {wR:11.3f}"
        )
        results[name] = r

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = np.arange(-1.3, 1.3001, 0.01)
    maps = {"truth (orbits within 14 A)": c_true}
    for name in ("(h,k,1) plane only", "streak lines + 2% background"):
        maps[f"refined: {name}"] = results[name].c
    fig, ax = plt.subplots(1, len(maps), figsize=(6 * len(maps), 5.5))
    ref = None
    for a, (title, c) in zip(ax, maps.items()):
        pdf = delta_pdf_slab(
            orbit_pair_terms(orbits, active, c), structure, x, x, (-0.1, 0.1)
        )
        rr = np.hypot(*np.meshgrid(x, x))
        ref = np.abs(pdf[rr > 0.15]).max() if ref is None else ref
        a.imshow(
            pdf / ref,
            origin="lower",
            extent=[x[0], x[-1], x[0], x[-1]],
            cmap="bwr",
            vmin=-1,
            vmax=1,
        )
        a.set_title(title)
        a.set_xlabel("[dx, 0, 0]")
    ax[0].set_ylabel("[0, dy, 0]")
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "tb5ru6sn18_recovery_delta_pdf.png", dpi=80)


if __name__ == "__main__":
    main()
