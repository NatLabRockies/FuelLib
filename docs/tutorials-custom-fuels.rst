Adding Custom Fuels
====================

This tutorial explains how to add custom fuels to FuelLib. Custom fuels allow you to use your own fuel composition and property data with the FuelLib calculations and plotting tools.

Directory Structure
-------------------

Create a fuel data directory with this structure:

.. code-block:: text

    customFuels/
    ├── gcData/
    │   └── your_fuel_name_init.csv
    ├── groupDecompositionData/
    │   └── your_fuel_name.csv
    └── fuel_metadata.yaml

**Required subdirectories:**

- ``gcData/``: Contains GC×GC composition data (one file per fuel)
- ``groupDecompositionData/``: Contains functional group decomposition data (one file per fuel)

**Required file:**

- ``fuel_metadata.yaml``: Configuration file that maps fuel names to their decomposition files

Metadata Configuration
----------------------

Each custom fuel directory must have a ``fuel_metadata.yaml`` file at the root of the ``fuelData`` directory. This file defines the mapping from fuel names to their group decomposition files.

**Minimal format:**

.. code-block:: yaml

    fuels:
      your_fuel:
        decomp_name: your_fuel

**Fields:**

- ``decomp_name`` (required): Name of the decomposition CSV file (without ``.csv`` extension). Must match a file in ``groupDecompositionData/``

**Optional fields:**

You can add additional metadata for documentation:

.. code-block:: yaml

    fuels:
      your_fuel:
        name: Display Name for Your Fuel
        category: Conventional|SATF|Simple
        source: Citation or origin of fuel data
        reference: URL to source paper
        description: Brief description of the fuel
        decomp_name: name_of_decomposition_file in ``groupDecompositionData/`` (without ``.csv`` extension)
        props_data: Name of any related properties data file in ``propertiesData/`` (without ``.csv`` extension)


Note that you can assign the same decomposition to multiple fuel variants if they have identical bulk composition.

GCxGC Composition Data
----------------------

Create a file named ``{fuel_name}_init.csv`` in the ``gcData/`` directory with fuel composition data.

**Required columns:**

- ``Compound``: Name of each component
- ``Weight %``: Weight percentage of each component

**Example:**

.. code-block:: text

    Compound,Weight %
    n-Decane,60
    n-Dodecane,40

Group Decomposition Data
------------------------

Create a file named ``{decomp_name}.csv`` in the ``groupDecompositionData/`` directory with functional group decompositions for each compound.

See the `Basic Usage tutorial <tutorials-basic.html#decomposing-fuel-components-into-fundamental-groups>`_ for detailed information on group decompositions.


Using Custom Fuels
------------------

Once your custom fuel directory is set up, you can use it like any built-in fuel by specifying the ``fuelDataDir`` when creating a fuel object:

.. code-block:: python

    import fuellib as fl

    # Load a custom fuel
    fuel = fl.fuel("new-satf", fuelDataDir="/path/to/customFuels")

    # Calculate the saturated vapor pressure at 320 K
    T = 320  # K
    p_sat_i = fuel.psat(T)
    p_sat_mix = fuel.mixture_vapor_pressure(fuel.Y_0, T)

Tips and Best Practices
-----------------------

1. **Composition Normalization**: Weight percentages don't need to sum to exactly 100% - FuelLib normalizes them automatically.

2. **Group Decomposition Accuracy**: Predictions depend heavily on decomposition quality. When possible you should validate individual compound properties against measured properties or NIST WebBook.

3. **Fuel Variants**: Use ``decomp_name`` to map multiple fuel variants to the same decomposition file when they have identical compounds but different weight percentages.

   .. code-block:: yaml

       fuels:
         fuel_1:
           decomp_name: fuel_decomp_1_and_2
         
         fuel_2:
           decomp_name: fuel_decomp_1_and_2
