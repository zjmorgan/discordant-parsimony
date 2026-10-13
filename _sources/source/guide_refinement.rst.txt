Refinement, Validation, and Participation
=========================================

Covers how refinable correlations are fitted to measured intensities:
the design matrix, choosing reciprocal-space positions, the
least-squares problem and its uncertainties, validation on held-out
data, and the participation scan that tests which atoms are correlated.
Complete data are not required; what matters is a 3D sampling that
determines the coefficients.

Design matrix
-------------
The intensity is linear in the orbit coefficients, :math:`I = A
\mathbf{c}`. Column :math:`k` of
:func:`~discordant_parsimony.scattering.numpy_backend.design_matrix` is

.. math::

   A_k(\mathbf{Q}) = \sum_{p \in k} b_i b_j\, e^{-W_i - W_j}\,
   \cos(2\pi\,\mathbf{h}\cdot\mathbf{d}_p)
   \prod_a \operatorname{sinc}(\Delta h_a\, d_{p,a}),

with bin averaging included when bin widths are given. It is computed
once per selection of positions and reused by every refinement and
model comparison.

Selecting positions
-------------------
:mod:`~discordant_parsimony.selection` reduces a volume to the positions
that carry independent diffuse information:

- **Bragg mask.** Reflections allowed by the lattice centering (e.g.
  all-even or all-odd for *F*) are excluded within a radius in
  Å\ :sup:`-1`. The mask is a separate selection, so Bragg regions can
  be reintroduced for a later joint Bragg and diffuse refinement.
- **Laue asymmetric unit.** Symmetrized data repeat each value across
  equivalent positions; keeping one representative per orbit
  (:func:`~discordant_parsimony.selection.canonical`) avoids counting
  copies as independent observations.
- **Satellites and lines.** Diffuse maxima are targeted like
  satellites: points :math:`\mathbf{G} + \mathbf{k}` or lines
  :math:`\mathbf{G} + \mathbf{k} + t\mathbf{v}` for allowed
  :math:`\mathbf{G}` and the Laue star of :math:`\mathbf{k}` and
  :math:`\mathbf{v}`.
- **Background sampling.** A reproducible random fraction of the
  remaining positions, recorded with its seed.

Least squares
-------------
With observed intensities :math:`y`, uncertainties :math:`\sigma`, a
scale :math:`s`, and a smooth background basis :math:`B`,

.. math::

   y = s\,A(\mathbf{c}_0 + N\boldsymbol{\theta}) + B\boldsymbol{\beta}
     = s\,A\mathbf{c}_0 + (AN)\,\mathbf{u} + B\boldsymbol{\beta},
   \qquad \mathbf{u} = s\boldsymbol{\theta}.

The problem is linear in :math:`(s, \mathbf{u}, \boldsymbol{\beta})`
and is solved in one weighted step by
:func:`~discordant_parsimony.refine.refine`. The absolute scale comes
from the fixed within-cluster terms in :math:`\mathbf{c}_0`.
Uncertainties of every coefficient follow from the covariance of the
solution by the delta method. The numerical rank and singular values
are reported, and an optional ridge damping handles poorly covered
selections. Tests confirm exact recovery from noise-free data and that
the reported uncertainties match the scatter over repeated noisy
refinements.

Validation on held-out data
---------------------------
Refinements are scored on positions that were not used to fit them:
the reduced :math:`\chi^2` and the weighted R factor on the held-out
set, playing the role of :math:`R_\text{free}`. A model with more
parameters gains nothing unless it predicts unseen intensities better.

Participation scan
------------------
:func:`~discordant_parsimony.participation.participation_scan` answers
which atoms are correlated. Candidate units are sets of site labels,
usually one split-site cluster type each. For every combination of
units, correlations involving units that are switched off are set to
zero, while the exact within-cluster values are kept. Each model is
refined on a training set and ranked by its held-out :math:`\chi^2`.
Because the design matrix is shared, the scan costs one small
singular-value decomposition per model.

Real-space maps from refined coefficients
-----------------------------------------
Refined coefficients give a model 3D-:math:`\Delta`\ PDF through
:func:`~discordant_parsimony.symmetry.orbit_pair_terms` and
:func:`~discordant_parsimony.realspace.delta_pdf_slab`. Unlike a direct
Fourier inversion of the data, the map has no Bragg punch-and-fill,
coverage gaps, or termination ripples, and needs only data that
determine the coefficients. Individual coefficients can be less well
determined than the map itself, because errors in combinations that
share a real-space position cancel there.

Sampling in practice
--------------------
In the synthetic tests of :doc:`tutorial_tb5ru6sn18`, a random 10% of
the independent positions, or the streak lines plus a 2% background
sample, recover the 3D-:math:`\Delta`\ PDF and predict held-out
intensities at the noise level. A single plane cannot determine 3D
correlations, and the feature lines alone, without background
positions, leave the problem rank deficient.
