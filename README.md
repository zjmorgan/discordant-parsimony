# discordant-parsimony

Documentation: https://zjmorgan.github.io/discordant-parsimony

Forward modeling of single-crystal diffuse scattering directly from
finite-range pair correlations. Working name; successor to
[rmc-discord](https://github.com/zjmorgan/rmc-discord). No atomistic
supercell, Hamiltonian, or reverse Monte Carlo is needed to evaluate a
model.

## Model

Every nuclear model reduces to weighted Gaussian pair terms
`(d, w, U)`:

    I(Q) = sum_p  w_p exp(i Q.d_p) exp(-Q^T U_p Q / 2)

which is the Fourier transform of a 3D-ΔPDF built from Gaussians.
Occupational disorder on split sites, discrete displacements, and
uncorrelated thermal motion all take this form, so the engine does not
distinguish them. Models differ only in how they generate the weights:
local occupation rules (split-site clusters, forbidden contacts) are
derived from the average structure in `rules.py`.

## Conventions

* `h` in r.l.u. of the CIF cell; `Q = 2 pi B h` with `B = inv(A).T`,
  where the columns of `A` are the direct basis vectors (Angstrom).
* Fourier sign `exp(+i Q.r)`; pair vector `d = R + x_j - x_i` for
  covariance `C_ij(R) = <dn_i(0) dn_j(R)>`.
* Intensity in fm^2 per unit cell (neutron scattering lengths).
* `U` are Cartesian covariance tensors in Angstrom^2.
* Histogram bins are averaged exactly with `PairTerms.bin_average`.

## Data

Experimental data are never committed. Put them in `local_data/` at the
repository root (ignored) or set `DIFFUSE_DATA_ROOT`. Tests that need
private data are skipped when it is absent.

## Development

    conda env create -f environment.yml
    conda activate discordant
    python -m pip install -e .
    pytest

Build the documentation locally:

    sphinx-apidoc -o docs/source src/discordant_parsimony
    sphinx-build -b html docs docs/_build/html
