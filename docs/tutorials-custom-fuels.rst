Adding Custom Fuels
====================

This tutorial explains how to add custom fuels to FuelLib. Custom fuels allow you to use your own fuel composition and property data with the FuelLib calculations and plotting tools.

Directory Structure
-------------------

A user database mirrors the layout of the FuelLib database in ``fuellib/database``:

.. code-block:: text

    customFuels/
    ├── gcData/
    │   └── your_fuel_name.csv
    ├── referenceCompounds/        (optional)
    │   ├── compounds.csv
    │   └── gani.csv
    ├── propertiesData/            (optional)
    │   └── your_fuel_name.csv
    └── fuel_metadata.yaml         (optional)

A blank template with the CSV headers filled in can be created with
:func:`~fuellib.database.database.write_template`:

.. code-block:: python

    import fuellib as fl

    fl.database.write_template("customFuels", "your_fuel_name")

**Required subdirectories:**

- ``gcData/``: Contains GC×GC composition data (one file per fuel)

**Optional subdirectories:**

- ``referenceCompounds/``: Reference compounds that are not in FuelLib (or that override FuelLib compounds)
- ``propertiesData/``: Measured fuel properties for validation and plotting

Files in the user database take precedence over FuelLib files with the same name, and
user reference compounds replace FuelLib reference compounds with the same InChI or
common name.

GCxGC Composition Data
----------------------

Create a file named ``{fuel_name}.csv`` in the ``gcData/`` directory with fuel composition data.

**Required columns:**

- ``SMILES`` and/or ``Common_Name``: Identifies the reference compound of each component.
  Components are matched by SMILES (via InChI) when given, otherwise by common name
  (case-insensitive).
- ``Weight %``: Weight percentage of each component

**Optional columns:**

- ``PelePhysics_Key``: Species name of each component in a PelePhysics mechanism

**Example:**

.. code-block:: text

    Common_Name,Weight %
    n-decane,60
    n-dodecane,40

Reference Compounds
-------------------

Every component must match a reference compound in FuelLib or in the user
``referenceCompounds/compounds.csv``. To add a compound, only ``Common_Name`` and ``SMILES``
are required; the ``Family``, ``Num_C``, and ``InChI`` columns are populated from the
SMILES when the fuel is loaded. Literature values can be provided for any property along
with its ``_units``, ``_err``, and ``_source`` columns (e.g. ``Tc``, ``Tc_units``,
``Tc_err``, ``Tc_source``).

Missing properties that the Constantinou-Gani method can predict are populated from the
group decomposition of the compound in ``referenceCompounds/gani.csv`` (keyed by InChI).
The populated values are written back to the user ``compounds.csv``. See the
`Basic Usage tutorial <tutorials-basic.html#decomposing-fuel-components-into-fundamental-groups>`_
for detailed information on group decompositions.

Metadata Configuration
----------------------

An optional ``fuel_metadata.yaml`` file at the root of the directory documents the source
of each fuel and is displayed by ``fl-fuels -dir customFuels -v``:

.. code-block:: yaml

    fuels:
      your_fuel:
        name: Display Name for Your Fuel
        category: Conventional|SATF|Simple
        source: Citation or origin of fuel data
        reference: URL to source paper
        description: Brief description of the fuel

Using Custom Fuels
------------------

Once your custom fuel directory is set up, you can use it like any built-in fuel by specifying the ``userDataDir`` when creating a fuel object:

.. code-block:: python

    import fuellib as fl

    # Load a custom fuel
    fuel = fl.Fuel("new-saf", userDataDir="/path/to/customFuels")

    # Calculate the saturated vapor pressure at 320 K
    T = fl.Units.Quantity(320, "K") # Temperature as a pint.Quantity
    p_sat_i = fuel.psat(T)
    p_sat_mix = fuel.mixture_vapor_pressure(fuel.Y_0, T)

Tips and Best Practices
-----------------------

1. **Composition Normalization**: Weight percentages don't need to sum to exactly 100% - FuelLib normalizes them automatically.

2. **Group Decomposition Accuracy**: Predictions depend heavily on decomposition quality. When possible you should validate individual compound properties against measured properties or NIST WebBook, and provide literature values in ``compounds.csv``.

3. **Fuel Variants**: Fuels with the same compounds but different weight percentages share the same reference compounds, so only a new ``gcData`` file is needed for each variant.
