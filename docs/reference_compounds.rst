Reference Compounds
===================

It is difficult to identify individual components of complex multicomponent jet fuels using GCxGC analysis, 
which generally provides weight percentages of a given hydrocarbon family and carbon number within a sample (e.g., 5% C10 iso-alkane, 2% C13 cycloalkane, etc.).
To address this challenge, FuelLib uses a set of reference compounds that represent the major hydrocarbon families and carbon numbers found in jet fuels.
A comprehensive list of the reference compounds used in FuelLib can be found in the 
`fuellib/data/ref_compounds.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/data/ref_compounds.csv>`_ file, 
with associated Constantinou-Gani\ :footcite:p:`constantinou_new_1994,constantinou_estimation_1995` functional group decompositions in `fuellib/data/ref_gani.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/data/ref_gani.csv>`_,
Nannoolal\ :footcite:p:`nannoolal_estimation_2009` group decompositions in `fuellib/data/ref_nannoolal.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/data/ref_nannoolal.csv>`_,
and UNIFAC 2.0\ :footcite:p:`hayer_advancing_2025` subgroup decompositions in `fuellib/data/ref_unifac.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/data/ref_unifac.csv>`_.

The table below lists the family, carbon number, name, and SMILES string of each reference compound. 
The measured property columns of ``ref_compounds.csv`` are omitted here; see :ref:`sec-reference-properties` 
for a description of those data.

.. csv-table:: Reference compounds by GCxGC bin.
   :file: generated/ref_compounds_table.csv
   :header-rows: 1
   :align: center
   :widths: auto

.. _sec-reference-properties:

Reference Property Data
-----------------------

Measured properties of pure compounds can replace group-contribution predictions.
These data are stored directly in ``fuellib/data/ref_compounds.csv``. Each compound has the columns
``Family``, ``Carbon Number``, ``Reference Compound``, and ``SMILES``, followed by a block of five columns
for each property: ``{Property}_Value``, ``{Property}_Error``, ``{Property}_Units``, ``{Property}_Source``,
and ``{Property}_Notes``. Blank values indicate that no reference data are available for that compound.

``Family`` is one of ``n-alkane``, ``isoalkane``, ``alkene``, ``monocycloalkane``, ``dicycloalkane``,
``tricycloalkane``, ``alkylbenzene``, ``cycloaromatic``, or ``diaromatic``. The following properties are
supported:

=============  ====================================================  =====================================================
Property       Attribute                                             Behavior
=============  ====================================================  =====================================================
``Tc``         :attr:`~fuellib.fuel.components.Components.tc`        Reference value, else group-contribution prediction
``Pc``         :attr:`~fuellib.fuel.components.Components.pc`        Reference value, else group-contribution prediction
``Vc``         :attr:`~fuellib.fuel.components.Components.vc`        Reference value, else group-contribution prediction
``Tm``         :attr:`~fuellib.fuel.components.Components.tm`        Reference value, else group-contribution prediction
``Tb``         :attr:`~fuellib.fuel.components.Components.tb`        Reference value, else group-contribution prediction
``Hf_STP``     :attr:`~fuellib.fuel.components.Components.hf_stp`    Reference value, else group-contribution prediction
``Hv_STP``     :attr:`~fuellib.fuel.components.Components.hv_stp`    Reference value, else group-contribution prediction
``Vm_STP``     :attr:`~fuellib.fuel.components.Components.vm_stp`    Reference value, else group-contribution prediction
``Omega``      :attr:`~fuellib.fuel.components.Components.omega`     Reference value, else group-contribution prediction
``YSI``        :attr:`~fuellib.fuel.components.Components.ysi`       Reference only; raises ``ValueError`` if missing
``DCN``        :attr:`~fuellib.fuel.components.Components.dcn`       Reference only; raises ``ValueError`` if missing
=============  ====================================================  =====================================================

Reference values are converted from the stated ``{Property}_Units`` to the attribute's units. Measured
values are used for properties in the first group on a compound-by-compound basis, so a single fuel can mix
reference values and predictions. Pass ``use_ref_props=False`` to :class:`~fuellib.fuel.core.Fuel` to use
only the group-contribution predictions. ``YSI`` (Yield Sooting Index) and ``DCN`` (derived cetane number)
are not predicted by FuelLib, so they are always taken from the reference data.

Each GCxGC row is matched to a reference compound by ``Reference Compound`` name or by the InChI derived
from its ``SMILES``; the matched rows are available as ``Fuel.components_data``, and the
group decompositions from ``ref_gani.csv``, ``ref_nannoolal.csv``, and ``ref_unifac.csv`` as
``Fuel._gani_decomp``, ``Fuel._nannoolal_decomp``, and
``Fuel._unifac_decomp``. Custom tables are read from ``ref_compounds.csv``, ``ref_gani.csv``,
``ref_nannoolal.csv``, and ``ref_unifac.csv`` in the ``data_dir`` passed to :class:`~fuellib.fuel.core.Fuel` (see :doc:`tutorials_custom_fuels`). Each custom table is merged
with, rather than replacing, the bundled table: unique custom rows are added, while duplicates take precedence over
the bundled rows. A reference-compound duplicate is identified when its SMILES resolves to the same InChI as a
bundled compound; a Gani, Nannoolal, or UNIFAC decomposition duplicate is identified by its ``Reference Compound``.

References
----------

.. footbibliography::
