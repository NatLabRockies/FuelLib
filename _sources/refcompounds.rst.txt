Reference Compounds
===================

It is difficult to identify individual components of complex multicomponent jet fuels using GCxGC analysis, 
which generally provides weight percentages of a given hydrocarbon family and carbon number within a sample (e.g., 5% C10 iso-alkane, 2% C13 cycloalkane, etc.).
To address this challenge, FuelLib uses a set of reference compounds that represent the major hydrocarbon families and carbon numbers found in jet fuels.
A comprehensive list of the reference compounds used in FuelLib can be found in the 
`fuellib/data/refCompounds.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/data/refCompounds.csv>`_ file, 
with associated Constantinou-Gani\ :footcite:p:`constantinou_new_1994,constantinou_estimation_1995` functional group decompositions in `fuellib/data/refGani.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/data/refGani.csv>`_.

The table below lists the family, carbon number, name, and SMILES string of each reference compound. 
The measured property columns of ``refCompounds.csv`` are omitted here; see :ref:`sec-reference-properties` 
for a description of those data.

.. csv-table:: Reference compounds by GCxGC bin.
   :file: generated/refCompounds_table.csv
   :header-rows: 1
   :align: center
   :widths: auto

References
----------

.. footbibliography::
