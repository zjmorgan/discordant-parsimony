From the Average Structure to Refinable Correlations
====================================================

Covers how a CIF becomes a set of refinable parameters. Pairs are built
in the unit cell and extended to a distance cutoff. Symmetry reduces
them to one coefficient per orbit, and constraints that follow exactly
from the average structure remove what the data cannot or need not
determine. Nothing in this step assumes a model of the local order.

Participating positions
-----------------------
:meth:`~discordant_parsimony.structure.Structure.sites` expands the
asymmetric unit with the space-group operators, rotating anisotropic
displacement parameters onto each equivalent position. Occupational
correlations exist only where occupations fluctuate, so the candidates
are the partially occupied (or substituted) sites,
``sites.select(sites.partial())``.

Split-site clusters
-------------------
Positions closer than a cluster distance are alternatives: an atom and
its split partners, two species sharing a line, or several off-center
positions in a cage. :func:`~discordant_parsimony.rules.find_clusters`
groups them, checks that each group is a clique whose occupancies sum to
at most one, and records the vacancy probability. Coincident positions
of different labels (plain substitution) are included.

Forbidden contacts
------------------
:func:`~discordant_parsimony.rules.forbidden_contacts` lists pairs in
different clusters that are shorter than an allowed distance per
species pair. These are hypotheses about the local structure, not
counting identities, so they are not imposed during refinement: the
data test them.

Pair orbits
-----------
:func:`~discordant_parsimony.symmetry.pair_orbits` enumerates every pair
up to a distance cutoff, which, unlike a box of cell indices, is
invariant under the group. An operator :math:`(W, \mathbf{t})` maps
pair :math:`(i, j, \mathbf{R})` to :math:`(i', j', W\mathbf{R} +
\mathbf{L}_j - \mathbf{L}_i)`, where :math:`W\mathbf{x}_i + \mathbf{t}
= \mathbf{x}_{i'} + \mathbf{L}_i`; reversal maps it to
:math:`(j, i, -\mathbf{R})`. Correlations with the symmetry of the
average structure are equal on each orbit. This assumes that data
symmetrized to the Laue group reflect symmetric (domain-averaged)
correlations.

Exact constraints
-----------------
Three kinds of relation hold whatever the local order.

**Fixed within-cluster values.** Inside one cluster at most one member
is occupied, so :math:`C_{ab} = x_a \delta_{ab} - x_a x_b` for members
:math:`a, b` (self terms included). These are known from the average
structure, and holding them fixed puts the refined correlations on an
absolute scale (:func:`~discordant_parsimony.correlations.fixed_cluster_orbits`).

**Sum rules.** A cluster without vacancies always holds exactly one
atom, so :math:`\sum_a \delta n_a = 0` and its covariances with any
position sum to zero:

.. math::

   \sum_{a} C(a \,@\, \mathbf{o}_a,\; j \,@\, \mathbf{R}) = 0 .

For a two-member cluster this gives :math:`C(B, X) = -C(A, X)`: one
number per pair of clusters instead of four, and the reason a binary
alloy needs one Warren-Cowley parameter per shell. A six-position cage
keeps five independent directions. Clusters with vacancies get no rule
(:func:`~discordant_parsimony.constraints.sum_rule_rows`).

**Homometric ties.** Orbits whose pair vectors, scattering-length
products, and displacement tensors all coincide produce identical
intensities, so only their sum is measurable. They are tied equal, and
each then carries the group average
(:func:`~discordant_parsimony.constraints.homometric_groups`). This is
the diffuse counterpart of overlapping vectors in a Patterson map.

The constraints are solved once,
:math:`\mathbf{c} = \mathbf{c}_0 + N\boldsymbol{\theta}`, by
:func:`~discordant_parsimony.constraints.parameterize`. Any
:math:`\boldsymbol{\theta}` then satisfies every fixed value and rule
exactly.

Why the constraints matter
--------------------------
Without them, pairs that share a vector, such as A-A and B-B pairs
across a lattice translation, can be told apart only through their
scattering lengths and displacement parameters. The design matrix is
then nearly singular, and individual coefficients carry very large
uncertainties even from complete data. The sum rules remove exactly
these directions. In the Tb5Ru6Sn18 example (see
:doc:`tutorial_tb5ru6sn18`) they reduce 208 free coefficients within
14 Å to 89, and the refinement becomes full rank.
