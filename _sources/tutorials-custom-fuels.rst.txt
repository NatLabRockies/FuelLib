Adding Custom Fuels
====================

This tutorial explains how to add custom fuels to FuelLib. Custom fuels allow you to use your own fuel composition and property data with the FuelLib calculations and plotting tools.

Directory Structure
-------------------

Create a fuel data directory with this structure:

.. code-block:: text

    customFuels/
    ├── gcData/
    │   └── your_fuel_name.csv
    ├── propertiesData/          (optional)
    │   └── your_fuel_name.csv
    ├── refCompounds.csv         (optional)
    ├── refGani.csv              (optional)
    ├── refNannoolal.csv         (optional)
    └── fuel_metadata.yaml       (optional)

**Required:**

- ``gcData/{fuel_name}.csv``: GC×GC composition data (one file per fuel).

**Optional:**

- ``propertiesData/{fuel_name}.csv``: measured property data (one file per fuel) loaded into ``Fuel.propData``.
- ``refCompounds.csv``, ``refGani.csv``, and ``refNannoolal.csv``: custom reference compound and group decomposition tables. When present, each table supplements FuelLib's built-in table; it does not replace it. If a file is absent, FuelLib uses the built-in table in ``fuellib/data``.
- ``fuel_metadata.yaml``: display name, source, and other documentation for each fuel, keyed by fuel name under a top-level ``fuels`` entry. The entry for a fuel is loaded into ``Fuel.metadata`` (``None`` if the file or entry is absent).

Fuel variants with identical compounds but different weight percentages simply use separate ``gcData`` files that refer to the same reference compounds.

GCxGC Composition Data
----------------------

Create a file named ``{fuel_name}.csv`` in the ``gcData/`` directory with fuel composition data.

**Required columns:**

- ``Weight %``: Weight percentage of each component (normalized automatically).
- ``Reference Compound`` and/or ``SMILES``: Identifies the compound. Each row is matched to ``refCompounds.csv`` by ``Reference Compound`` name (case-insensitive) first, then by the InChI derived from ``SMILES``.

**Optional columns:**

- ``GC-Bin``: Name of the GCxGC bin, used as the compound name by the exporters when PelePhysics keys are not used.
- ``PelePhysics Key``: Species name of the compound in a PelePhysics mechanism.

**Example:**

.. code-block:: text

    GC-Bin,SMILES,PelePhysics Key,Weight %
    n-C10,CCCCCCCCCC,NC10H22,60
    n-C12,CCCCCCCCCCCC,NC12H26,40

A ``ValueError`` is raised if a row cannot be matched to a reference compound.

Reference Compounds and Group Decompositions
--------------------------------------------

Every compound in a fuel must appear in both effective reference tables. To add a new compound, add a row to each custom table:

- ``refCompounds.csv``: ``Family``, ``Carbon Number``, ``Reference Compound``, and ``SMILES`` are required. ``Family`` must be one of ``n-alkane``, ``isoalkane``, ``alkene``, ``monocycloalkane``, ``dicycloalkane``, ``tricycloalkane``, ``alkylbenzene``, ``cycloaromatic``, or ``diaromatic``. Measured properties are optional; see :ref:`sec-reference-properties`. A custom row replaces a built-in row when the two SMILES values resolve to the same InChI, so use this to override built-in properties even when the valid SMILES representation differs.
- ``refGani.csv``: the functional group decomposition of the compound, with a ``Reference Compound`` column matching ``refCompounds.csv``. A custom row with the same ``Reference Compound`` replaces the built-in decomposition. Groups that are omitted or blank are treated as zero.
- ``refNannoolal.csv``: the Nannoolal group decomposition of the compound, in the same format as ``refGani.csv`` but with the groups defined in `nannoolal.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/gcm/nannoolal.csv>`_.

Decompositions can be generated from SMILES with ``fl-decomp-smiles`` (``--method gani`` or ``--method nannoolal``; see :doc:`tutorials-cli`), but should be verified before use.

See the `Basic Usage tutorial <tutorials-basic.html#decomposing-fuel-components-into-fundamental-groups>`_ for detailed information on group decompositions.

Using Custom Fuels
------------------

Once your custom fuel directory is set up, you can use it like any built-in fuel by specifying the ``fuelDataDir`` when creating a fuel object. Custom reference tables are only read from ``fuelDataDir``, so they must follow the directory structure above:

.. code-block:: python

    import fuellib as fl

    # Load a custom fuel
    fuel = fl.Fuel("new-saf", fuelDataDir="/path/to/customFuels")

    # Calculate the saturated vapor pressure at 320 K
    T = fl.Units.Quantity(320, "K") # Temperature as a pint.Quantity
    p_sat_i = fuel.psat(T)
    p_sat_mix = fuel.mixture_vapor_pressure(fuel.Y_0, T)

Tips and Best Practices
-----------------------

1. **Composition Normalization**: Weight percentages don't need to sum to exactly 100% - FuelLib normalizes them automatically.

2. **Group Decomposition Accuracy**: Predictions depend heavily on decomposition quality. When possible you should validate individual compound properties against measured properties or NIST WebBook.

3. **Fuel Variants**: Create one ``gcData`` file per variant. Variants that share compounds also share the same reference table rows, so no additional decomposition data is needed.
