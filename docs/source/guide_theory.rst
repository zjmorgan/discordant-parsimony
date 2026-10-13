Pair Terms: One Model for Diffuse Scattering
============================================

Covers the scattering model that every other part of the package
builds on: diffuse intensity written as a sum of weighted Gaussian pair
terms, the conventions it uses, and why occupational disorder, split
sites, and discrete displacements need no separate treatment. The
model is linear in the pair correlations, which is what makes
refinement a crystallographic least-squares problem rather than an
atomistic simulation.

Conventions
-----------
Fractional coordinates :math:`\mathbf{x}` and Miller indices
:math:`\mathbf{h}` (r.l.u.) refer to the cell of the CIF. The direct
basis vectors are the columns of :math:`A` (Cartesian, Å, with
:math:`\mathbf{a}` along :math:`x` and :math:`\mathbf{b}` in the
:math:`xy` plane), and

.. math::

   B = (A^{-1})^{\mathsf{T}}, \qquad
   \mathbf{Q} = 2\pi B\mathbf{h}, \qquad
   \mathbf{Q}\cdot(A\mathbf{x}) = 2\pi\,\mathbf{h}\cdot\mathbf{x}.

The Fourier sign is :math:`e^{+i\mathbf{Q}\cdot\mathbf{r}}`. Intensities
are per unit cell in fm\ :sup:`2` (neutron scattering lengths), and
displacement parameters are Cartesian covariance tensors in
Å\ :sup:`2`.

The pair-term model
-------------------
Position :math:`i` in the origin cell and position :math:`j` in cell
:math:`\mathbf{R}` are separated by

.. math::

   \mathbf{d}_{ij}(\mathbf{R}) = \mathbf{R} + \mathbf{x}_j - \mathbf{x}_i .

Every model is reduced to terms :math:`(\mathbf{d}_p, w_p, U_p)`:

.. math::

   I(\mathbf{Q}) = \sum_p w_p\,
   \cos(2\pi\,\mathbf{h}\cdot\mathbf{d}_p)\,
   \exp\!\left(-\tfrac{1}{2}\mathbf{Q}^{\mathsf{T}} U_p\,\mathbf{Q}\right).

The cosine replaces :math:`e^{i\mathbf{Q}\cdot\mathbf{d}}` because term
sets are closed under :math:`(\mathbf{d}, w, U) \to (-\mathbf{d}, w,
U)`, so the sine parts cancel exactly; ``evaluate(..., check=True)``
verifies this. In real space the same terms are a 3D-:math:`\Delta`\ PDF
made of Gaussians,

.. math::

   \Delta P(\mathbf{r}) = \sum_p w_p\, G(\mathbf{r} - \mathbf{d}_p;\, U_p),

so a model predicts the intensity and the :math:`\Delta`\ PDF from the
same numbers (:func:`~discordant_parsimony.realspace.delta_pdf_slab`).

Occupational and split-site disorder
------------------------------------
With occupation :math:`n_i \in \{0, 1\}`, average :math:`x_i`, and
fluctuation :math:`\delta n_i = n_i - x_i`, the covariance

.. math::

   C_{ij}(\mathbf{R}) = \langle \delta n_i(\mathbf{0})\,
   \delta n_j(\mathbf{R}) \rangle

gives terms with :math:`w = b_i b_j C_{ij}(\mathbf{R})` at
:math:`\mathbf{d}_{ij}(\mathbf{R})`. Uncorrelated Gaussian thermal
motion smears each term by :math:`U = U_i + U_j`, i.e. multiplies it by
the Debye-Waller factors :math:`e^{-W_i - W_j}` with
:math:`W_i = \tfrac{1}{2}\mathbf{Q}^{\mathsf{T}} U_i \mathbf{Q}`. The
covariance satisfies :math:`C_{ij}(\mathbf{R}) = C_{ji}(-\mathbf{R})`,
which is the reversal closure above.

Discrete displacements are written as split sites: an atom that sits
in one of several nearby positions is a set of alternative positions,
at most one occupied. The model is then exact for any displacement
size. No expansion in :math:`\mathbf{Q}\cdot\mathbf{u}` is made, which
matters because such expansions fail at the large :math:`|\mathbf{Q}|`
of time-of-flight data.

Uncorrelated thermal motion also leaves an incoherent Laue term
:math:`x_i b_i^2 (1 - e^{-2W_i})` per position, available as
:func:`~discordant_parsimony.pairs.thermal_laue_terms` and stored as two
terms at :math:`\mathbf{d} = 0`.

Exact histogram-bin averaging
-----------------------------
Measured volumes are histograms, not point samples. Averaging
:math:`e^{2\pi i \mathbf{h}\cdot\mathbf{d}}` over a bin of widths
:math:`\Delta h_a` multiplies each term by a factor independent of
:math:`\mathbf{Q}`:

.. math::

   w_p \;\to\; w_p \prod_a \operatorname{sinc}(\Delta h_a\, d_{p,a}),
   \qquad \operatorname{sinc}(x) = \frac{\sin \pi x}{\pi x}.

This is exact for unsmeared terms and neglects only the variation of
the Gaussian factor across one bin otherwise. On a 0.25 r.l.u. grid the
factor is 0.90 at one cell and 0.64 at two, so it is not negligible.

What the data can resolve
-------------------------
On a grid with spacing :math:`1/N` r.l.u., pair vectors that differ by
:math:`N` lattice translations give identical phases at every grid
point. A 0.25 r.l.u. grid therefore resolves correlations within about
:math:`\pm 2` cells. Beyond that range a pair-correlation model has
nothing to fit, which sets a natural upper bound on the cutoff.

Physical validity
-----------------
A non-negative intensity is necessary but not sufficient. A valid
covariance has a Hermitian positive semidefinite site matrix
:math:`\tilde C_{ij}(\mathbf{q}) = \sum_{\mathbf{R}} C_{ij}(\mathbf{R})
e^{i\mathbf{q}\cdot\mathbf{R}}` at every :math:`\mathbf{q}`, and 0/1
occupations add bounds such as
:math:`\max(0, x_i + x_j - 1) \le \langle n_i n_j \rangle \le
\min(x_i, x_j)`. Truncating a correlation function at a finite cutoff
is the usual way to violate these conditions.

Verification
------------
The test suite checks the model against brute force. For periodic
configurations, the pair-sum intensity equals
:math:`\langle |F - \langle F \rangle|^2 \rangle / N` of the explicit
supercell to :math:`10^{-10}`, including Debye-Waller factors. The bin
average matches numerical integration over the bin, and the
slab-averaged :math:`\Delta`\ PDF matches 3D sampling of the Gaussians.

Not yet implemented
-------------------
Correlated displacements of fully occupied atoms (a pair-specific
:math:`U` and the occupation-displacement cross term), magnetic
scattering, and X-ray form factors :math:`f(Q)`. They fit the same
pair-term structure: correlated displacements change :math:`U_p`, and
the cross term adds sine-weighted terms.
