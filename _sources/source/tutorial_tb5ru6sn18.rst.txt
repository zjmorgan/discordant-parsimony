Tb5Ru6Sn18: Which Atoms Participate?
====================================

Walks through the full workflow on the Remeika phase Tb5Ru6Sn18
(*Fm*\ -3\ *m*, *a* = 13.7825 Å), measured on TOPAZ at 293 K [1]_. The
diffuse scattering is structural and comes from the partially occupied
Tb2, Sn1, and Sn2 sites. The experimental data are private and are not
distributed. The scripts in ``examples/`` reproduce every number below
when the data are available locally, and the synthetic results need no
experimental data beyond the CIF.

The average structure and its rules
-----------------------------------
.. code-block:: python

   from discordant_parsimony import Structure
   from discordant_parsimony.io import data_root
   from discordant_parsimony.rules import find_clusters, forbidden_contacts

   structure = Structure.from_cif(data_root() / "Tb5Ru6Sn18" / "Tb5Ru6Sn18.cif")
   sites = structure.sites()
   active = sites.select(sites.partial())  # Tb2, Sn1, Sn2: 112 positions
   clusters = find_clusters(structure, active, 1.2)
   limits = {("Sn", "Sn"): 2.6, ("Sn", "Tb"): 2.8, ("Tb", "Tb"): 3.0}
   forbidden = forbidden_contacts(structure, active, clusters, limits)
   for row in forbidden.summary(active):
       print(row)

The cell holds 188 positions, 112 of them partially occupied. They form
40 split-site clusters, none with vacancies:

- 32 Tb2/Sn1 lines along :math:`\langle 111 \rangle`: each line holds
  either Tb2 or Sn1, 1.09 Å apart.
- 8 Sn2 cages around 8c: one Sn2 in one of six off-centre positions,
  0.71 and 1.01 Å apart.

Two contacts between clusters are far shorter than chemically
reasonable: Sn1-Sn1 at 2.44 Å (three per Sn1) and Tb2-Sn2 at 2.51 Å.

Pairs, orbits, and constraints
------------------------------
.. code-block:: python

   import numpy as np
   from discordant_parsimony.symmetry import pair_orbits
   from discordant_parsimony.correlations import fixed_cluster_orbits
   from discordant_parsimony.constraints import (
       homometric_groups, parameterize, sum_rule_rows, tie_rows,
   )

   orbits = pair_orbits(structure, active, 14.0)
   fixed = fixed_cluster_orbits(orbits, active, clusters)
   group = homometric_groups(orbits, active)
   rows = np.vstack([sum_rule_rows(orbits, clusters), tie_rows(group, fixed[0])])
   p = parameterize(orbits.n_orbits, fixed, rows)

====================  ===========  ========================
Pair cutoff           Pairs/cell   Independent coefficients
====================  ===========  ========================
8 Å                   10 144       53
14 Å                  54 832       214
20 Å                  159 312      556
27.6 Å (~2 cells)     421 264      1 364
====================  ===========  ========================

Within 14 Å, six orbits are fixed by the average structure (three self
terms and the three within-cluster separations). The sum rules and
homometric ties then leave 89 free correlations instead of 208.

A local-order hypothesis for synthetic tests
--------------------------------------------
If both short contacts are forbidden, each 4b cube of Tb2/Sn1 lines
holds Sn1 on one tetrahedron of corners and Tb2 on the other, an Ising
pseudospin. Every 8c cage is surrounded by a tetrahedron of cubes, and
its single Sn2 fits only between two cubes that present Sn1. Counting
then forces exactly two Sn-facing cubes around every cage, a 2:2 rule
on every fcc tetrahedron (the ground state of the nearest-neighbour fcc
Ising antiferromagnet). The rule has no free parameters and reproduces
the stepped diffuse streaks of the (*h*, *k*, 1) plane and the signs of
the main features of the measured 3D-:math:`\Delta`\ PDF
(``examples/tb5ru6sn18_ice.py``, ``examples/tb5ru6sn18_delta_pdf.py``).

Here it serves only as a known truth for recovery tests; the refinement
itself does not assume it.

Recovery from incomplete data
-----------------------------
``examples/tb5ru6sn18_recovery.py`` generates noisy intensities (3%)
from the 2:2 correlations on the measured grid, restricted to the
independent positions with :math:`|h_a| \le 12` (20 085), and refines
them from subsets. Without the sum rules and ties:

==============================  ======  =========  ===============
Subset                          N       Rank       wR (held out)
==============================  ======  =========  ===============
complete                        20 085  210/211    0.056
random 10%                      1 979   210/211    0.060
random 2%                       438     210/211    0.092
streak lines + 2% background    2 161   210/211    0.090
streak lines only               1 826   102/211    1.40
(*h*, *k*, 1) plane only        1 131   205/211    fails
==============================  ======  =========  ===============

The 3D samplings predict held-out intensities at the noise level, and
the refined 3D-:math:`\Delta`\ PDF from the streak lines plus background
matches the truth. Individual coefficients were poorly determined
before the constraints were added; that analysis led to the sum rules
and homometric ties.

Participation scan
------------------
``examples/tb5ru6sn18_participation.py`` treats the two cluster types
as candidate units and refines every combination on a random 10% of the
independent positions (14 Å cutoff, quadratic background in
:math:`Q^2`), scoring them on the other 90%:

.. code-block:: python

   from discordant_parsimony.participation import participation_scan

   units = {"Tb2/Sn1 lines": ["Tb2", "Sn1"], "Sn2 cages": ["Sn2"]}
   models = participation_scan(y, sigma, A, orbits, active, units,
                               fixed, rows, train, B)

Synthetic truths with known answers:

==========================================  ==================  ==================
Truth                                       Best model          Held-out
                                                                :math:`\chi^2_\nu`
==========================================  ==================  ==================
2:2 states (lines and cages correlated)     lines + cages       1.06
lines correlated, cages independent         lines only          1.01
==========================================  ==================  ==================

In the second case, adding the uncorrelated cages costs 63 parameters
and gains nothing on held-out data.

Measured data, 293 K:

===========================  ==========  ==================  =======
Participating                Parameters  Held-out            wR
                                         :math:`\chi^2_\nu`
===========================  ==========  ==================  =======
lines + cages                93          5.53                0.122
lines only                   30          5.79                0.125
cages only                   31          7.46                0.142
none (independent clusters)  4           7.77                0.144
===========================  ==========  ==================  =======

The Tb2/Sn1 lines carry most of the correlated signal, and the Sn2
cages add a smaller, consistent improvement on held-out data. The fit
is not yet at the noise level (:math:`\chi^2_\nu \approx 5.5`).
Systematic contributions not modelled here, such as the Tb\ :sup:`3+`
paramagnetic :math:`f^2(Q)` background, thermal diffuse scattering,
Bragg tails beyond the mask, and correlations beyond 14 Å, mean the
formal significance is overstated. The ranking is a first result, not a
final one.

.. [1] A. D. Montero, G. T. McCandless, Z. J. Morgan, B. Schundelmier,
   C. M. Hoffmann, R. E. Baumbach, and J. Y. Chan, *Structurally
   Frustrated Remeika Phases: Tolerance Factor and Non-Kramers' Ion
   Effects*, Chem. Mater. **37**, 9014-9024 (2025).
   https://doi.org/10.1021/acs.chemmater.5c02568
