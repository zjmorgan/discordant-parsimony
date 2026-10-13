"""Tb5Ru6Sn18: model 3D-delta-PDF of the 2:2 pseudospin states.

Plots the (dx, dy, 0 +/- 0.1) slab in lattice units, the same cut as
the 293 K experimental 3D-delta-PDF. The model map is built directly
from the pair terms (no Bragg punch-and-fill or Q-range truncation),
so feature positions and signs are comparable, not ripples or scale.
"""

from pathlib import Path

import numpy as np

from discordant_parsimony import pair_terms_from_covariance
from discordant_parsimony.correlations import covariance_from_configurations
from discordant_parsimony.realspace import delta_pdf_slab

from tb5ru6sn18_ice import MAX_OFFSET, active, configurations, structure


def main():
    configs = configurations()
    R, C = covariance_from_configurations(configs, MAX_OFFSET)
    terms = pair_terms_from_covariance(active, R, C)
    terms = terms.prune(1e-12 * np.abs(terms.w).max())
    # attenuation of the measured map by its 0.25 r.l.u. histogram bins
    terms = terms.bin_average([0.25, 0.25, 0.25])

    x = np.arange(-1.3, 1.3001, 0.01)
    pdf = delta_pdf_slab(terms, structure, x, x, (-0.1, 0.1))
    r = np.hypot(*np.meshgrid(x, x))
    scale = np.abs(pdf[r > 0.15]).max()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(13, 6))
    extent = [x[0], x[-1], x[0], x[-1]]
    for a, lim in zip(ax, (1.3, 0.37)):
        im = a.imshow(
            pdf / scale,
            origin="lower",
            extent=extent,
            cmap="bwr",
            vmin=-1,
            vmax=1,
        )
        a.set_xlim(-lim, lim)
        a.set_ylim(-lim, lim)
        a.set_xlabel("[dx, 0, 0]")
        a.set_ylabel("[0, dy, 0]")
    ax[0].set_title("2:2 model 3D-dPDF (dx, dy, 0 +/- 0.1)")
    ax[1].set_title("inset")
    for label, (px, py) in {
        "(4)": (1, 0),
        "(5)": (0.5, 0.5),
    }.items():
        ax[0].annotate(label, (px, py), (px + 0.08, py + 0.08))
    fig.colorbar(im, ax=ax, label="model 3D-dPDF [scaled]")

    out = Path(__file__).resolve().parents[1] / "results"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "tb5ru6sn18_delta_pdf.png", dpi=90)

    for label, (px, py) in {"(4)": (1, 0), "(5)": (0.5, 0.5)}.items():
        i, j = np.argmin(abs(x - py)), np.argmin(abs(x - px))
        print(f"{label} at ({px}, {py}, 0): {pdf[i, j] / scale:+.3f}")


if __name__ == "__main__":
    main()
