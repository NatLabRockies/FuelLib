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
    └── fuel_metadata.yaml       (optional)

**Required:**

- ``gcData/{fuel_name}.csv``: GC×GC composition data (one file per fuel).

**Optional:**

- ``propertiesData/{fuel_name}.csv``: measured property data (one file per fuel) loaded into ``Fuel.propData``.
- ``refCompounds.csv`` and ``refGani.csv``: custom reference compound and group decomposition tables. If either file is not found in the directory, FuelLib prints a message and falls back to the built-in file in ``fuellib/data``.
- ``fuel_metadata.yaml``: display name, source, and other documentation for each fuel (used by ``fl-fuel-manager``). The :class:`~fuellib.fuel.Fuel` class does not read this file.

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

Every compound in a fuel must appear in both reference tables. To add a new compound, add a row to each:

- ``refCompounds.csv``: ``Family``, ``Carbon Number``, ``Reference Compound``, and ``SMILES`` are required. ``Family`` must be one of ``n-alkane``, ``isoalkane``, ``alkene``, ``monocycloalkane``, ``dicycloalkane``, ``tricycloalkane``, ``alkylbenzene``, ``cycloaromatic``, or ``diaromatic``. Measured properties are optional; see :ref:`sec-reference-properties`.
- ``refGani.csv``: the functional group decomposition of the compound, with ``Common_Name`` equal to the ``Reference Compound`` in ``refCompounds.csv``. Groups that are omitted or blank are treated as zero.

See the `Basic Usage tutorial <tutorials-basic.html#decomposing-fuel-components-into-fundamental-groups>`_ for detailed information on group decompositions.

Using Custom Fuels
------------------

Once your custom fuel directory is set up, you can use it like any built-in fuel by specifying the ``fuelDataDir`` when creating a fuel object. Custom reference tables can also be supplied directly with ``refCompoundsPath`` and ``refGaniPath``:

.. code-block:: python

    import fuellib as fl

    # Load a custom fuel
    fuel = fl.Fuel("new-saf", fuelDataDir="/path/to/customFuels")

    # Calculate the saturated vapor pressure at 320 K
    T = fl.Units.Quantity(320, "K") # Temperature as a pint.Quantity
    p_sat_i = fuel.psat(T)
    p_sat_mix = fuel.mixture_vapor_pressure(fuel.Y_0, T)

    # Use custom reference tables stored outside of the fuel data directory
    fuel = fl.Fuel(
        "new-saf",
        fuelDataDir="/path/to/customFuels",
        refCompoundsPath="/path/to/refCompounds.csv",
        refGaniPath="/path/to/refGani.csv",
    )

Tips and Best Practices
-----------------------

1. **Composition Normalization**: Weight percentages don't need to sum to exactly 100% - FuelLib normalizes them automatically.

2. **Group Decomposition Accuracy**: Predictions depend heavily on decomposition quality. When possible you should validate individual compound properties against measured properties or NIST WebBook.

3. **Fuel Variants**: Create one ``gcData`` file per variant. Variants that share compounds also share the same reference table rows, so no additional decomposition data is needed.
