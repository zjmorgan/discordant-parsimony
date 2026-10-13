Data Handling: Private Data, Formats, and Reduction
===================================================

Covers where experimental data live, which formats are read, and how a
normalised, background-subtracted volume is formed. Experimental data
and the products derived from them are never committed to the
repository.

Where data live
---------------
Put each dataset in its own folder, either under ``local_data/`` at the
repository root (ignored by git) or under the directory named by the
``DIFFUSE_DATA_ROOT`` environment variable;
:func:`~discordant_parsimony.io.data_root` resolves the location. Tests
that need private data are skipped when it is absent, so the suite runs
anywhere. Output of the examples goes to ``results/``, which is also
ignored. Only small synthetic fixtures are tracked.

Before committing, confirm that nothing private is staged:

.. code-block:: bash

   git status --short
   git ls-files | grep -Ei "local_data|results|\.nxs"

What a dataset needs
--------------------
- A CIF of the average structure with symmetry operators
  (``_space_group_symop_operation_xyz`` or
  ``_symmetry_equiv_pos_as_xyz``), occupancies, displacement parameters
  (``Uani`` or ``Uiso``), and type symbols.
- Volumes as Mantid MDHistoWorkspaces saved with ``SaveMD``: sample
  data and normalisation, and background data and normalisation.
- Whether the data are neutron or X-ray (only neutron is supported at
  present), the temperature, and the Laue group used to symmetrize the
  volume, if any.

Reading and reducing
--------------------
:func:`~discordant_parsimony.io.load_mdhisto` returns arrays indexed
along the workspace dimensions ``D0, D1, D2`` (the file stores them in
reverse order), the bin edges, and the projection matrix :math:`W`, with
Miller indices :math:`\mathbf{h} = W\mathbf{u}`.
:func:`~discordant_parsimony.io.reduce_volume` forms

.. math::

   I = \frac{\text{data}}{\text{norm}}
     - s\,\frac{\text{background data}}{\text{background norm}},

propagating uncorrelated uncertainties and marking bins with
non-positive normalisation as invalid.
