discordant-parsimony documentation
==================================

*discordant-parsimony* is a toolkit for the crystallography of diffuse
scattering. Starting from an average structure (CIF) and a measured
single-crystal diffuse volume, it determines which atoms take part in
correlated disorder and refines their pair correlations
:math:`C_{ij}(\mathbf{R})` directly, without an atomistic supercell,
Hamiltonian, or reverse Monte Carlo. It provides:

- A single forward model for occupational, split-site, and discretely
  displaced disorder: weighted Gaussian pair terms, equivalent to a
  3D-:math:`\Delta`\ PDF
- Local occupation rules derived from the average structure geometry
  (split-site clusters, forbidden contacts)
- Symmetry-independent pair correlations up to a distance cutoff, with
  exact constraints (sum rules, homometric ties, fixed within-cluster
  values)
- Crystallographic selection of reciprocal-space positions (Bragg
  masking, Laue asymmetric unit, satellite points and lines, background
  sampling)
- Linear least-squares refinement with absolute scale, uncertainties,
  and held-out validation
- A participation scan that tests which atoms are correlated
- Model 3D-:math:`\Delta`\ PDF maps from refined coefficients, free of
  Fourier truncation and punch-and-fill artifacts, from incomplete data

It is a successor to `rmc-discord <https://github.com/zjmorgan/rmc-discord>`_.

Core libraries and technologies:

- Python 3
- NumPy, SciPy for scientific computation
- PyCifRW for reading crystallographic information files
- h5py for Mantid MDHisto volumes
- Matplotlib for plotting
- Sphinx for documentation

Experimental data are never stored in the repository; see
:doc:`source/guide_data`.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   source/discordant_parsimony
   source/discordant_parsimony.guides
   source/discordant_parsimony.tutorials
