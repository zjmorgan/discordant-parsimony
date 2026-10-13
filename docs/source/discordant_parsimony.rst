discordant_parsimony package
============================

*discordant_parsimony* models single-crystal diffuse scattering from
finite-range pair correlations and refines those correlations against
measured intensities.

Features
--------
- Average structure from CIF with symmetry expansion of anisotropic
  displacement parameters
- Split-site clusters and forbidden contacts from interatomic distances
- Pair orbits of the space group within a distance cutoff
- Design matrix of the intensity per unit correlation, with exact
  histogram-bin averaging
- Constrained linear refinement and participation scans
- Real-space 3D-:math:`\Delta`\ PDF of any set of pair terms

Subpackages
-----------

.. toctree::
   :maxdepth: 4

   discordant_parsimony.scattering
   discordant_parsimony.tables

Workflow modules
----------------

discordant_parsimony.structure module
-------------------------------------

.. automodule:: discordant_parsimony.structure
   :members:
   :undoc-members:
   :show-inheritance:

discordant_parsimony.rules module
---------------------------------

.. automodule:: discordant_parsimony.rules
   :members:
   :undoc-members:
   :show-inheritance:

discordant_parsimony.symmetry module
------------------------------------

.. automodule:: discordant_parsimony.symmetry
   :members:
   :undoc-members:
   :show-inheritance:

discordant_parsimony.correlations module
----------------------------------------

.. automodule:: discordant_parsimony.correlations
   :members:
   :undoc-members:
   :show-inheritance:

discordant_parsimony.constraints module
---------------------------------------

.. automodule:: discordant_parsimony.constraints
   :members:
   :undoc-members:
   :show-inheritance:

discordant_parsimony.pairs module
---------------------------------

.. automodule:: discordant_parsimony.pairs
   :members:
   :undoc-members:
   :show-inheritance:

discordant_parsimony.selection module
-------------------------------------

.. automodule:: discordant_parsimony.selection
   :members:
   :undoc-members:
   :show-inheritance:

discordant_parsimony.refine module
----------------------------------

.. automodule:: discordant_parsimony.refine
   :members:
   :undoc-members:
   :show-inheritance:

discordant_parsimony.participation module
-----------------------------------------

.. automodule:: discordant_parsimony.participation
   :members:
   :undoc-members:
   :show-inheritance:

discordant_parsimony.realspace module
-------------------------------------

.. automodule:: discordant_parsimony.realspace
   :members:
   :undoc-members:
   :show-inheritance:

discordant_parsimony.io module
------------------------------

.. automodule:: discordant_parsimony.io
   :members:
   :undoc-members:
   :show-inheritance:
