Introduction and Basic Usage
----------------------------

FuelLib is a Python library that uses group contribution methods to estimate the 
thermophysical properties of multicomponent liquid fuels and their mixtures. This tutorial 
provides a basic introduction to using FuelLib, including installation, required input files, and example usage.

Download and Setup
^^^^^^^^^^^^^^^^^^

Clone the FuelLib repository from GitHub: ::

    git clone https://github.com/NatLabRockies/FuelLib.git

Create and activate a Conda environment, install the required dependencies: ::

    conda create --name fuellib-env matplotlib pandas scipy
    conda activate fuellib-env

Change to the FuelLib/tutorials directory: ::

    cd FuelLib/tutorials

Required Input Files and Decomposing Fuel Components
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

FuelLib requires two input files for any given fuel, ``<fuel_name>``:

- ``FuelLib/fuelData/gcData/<fuel_name>_init.csv``: the initial weight percentage composition of the fuel components (must include columns "Compound" and "Weight %")
- ``FuelLib/fuelData/groupDecompositionData/<fuel_name>.csv``: the fundamental group decomposition for each component of the fuel (must have columns for groups in the same order as `gcmTable <https://github.com/NatLabRockies/FuelLib/blob/main/gcmTableData/gcmTable.csv>`_)

These two required files must have the same number of rows and the same order of components. Many examples can be found in the `fuelData <https://github.com/NatLabRockies/FuelLib/tree/main/fuelData>`_ directory.

Decomposing Fuel Components into Fundamental Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The decomposition process is a manual process that requires knowledge of the chemical structure of each component in the fuel. 
For example, consider a multicomponent fuel consisting of heptane (C7H16), decane (C10H22), and (1,5-dimethylhexyl)cyclohexane (C14H28).

.. figure:: /figures/decomp-example.png
   :width: 300pt
   :align: center
   
   Chemical structures of heptane (top), decane (middle), and (1,5-dimethylhexyl)cyclohexane (bottom). The CH3 groups are highlighted in green, the CH2 groups in blue, and the CH group in gold. The additional second order structures of (1,5-dimethylhexyl)cyclohexane are shown in orange.

The decomposition of heptane and decane into the groups defined in the `gcmTable`_ is straightforward, as both compounds consist of only CH3 and CH2 groups. 
(1,5-dimethylhexyl)cyclohexane is a branched cycloalkane with a six-membered ring and 
two methyl branches on the first and fifth carbon atoms of the ring, as shown in the figure above.
The six-membered ring and the branch containing the two CH3 groups bonded to a CH group 
are accounted for using the second order groups defined in the `gcmTable`_. However, 
the remaining branch with a single CH3 group bonded to a CH2 group is not defined in the `gcmTable`_.

.. table::
    :align: center

    +--------------------------------+-------+-------+-------+-----+----------+-----------------+
    | Compound                       | CH3   | CH2   | CH    | ... | (CH3)2CH | 6 membered ring |
    +================================+=======+=======+=======+=====+==========+=================+
    | Heptane                        | 2     | 5     | 0     | ... | 0        | 0               |
    +--------------------------------+-------+-------+-------+-----+----------+-----------------+
    | Decane                         | 2     | 8     | 0     | ... | 0        | 0               |            
    +--------------------------------+-------+-------+-------+-----+----------+-----------------+
    | (1,5-dimethylhexyl)cyclohexane | 3     | 8     | 3     | ... | 1        | 1               |
    +--------------------------------+-------+-------+-------+-----+----------+-----------------+

Basic Usage
^^^^^^^^^^^

To demonstrate the usage of FuelLib, we will use the fuel "heptane-decane", which is a 
binary mixture of heptane and decane. The initial weight percentage composition is 73.75% 
heptane and 26.25% decane, and the group decomposition data is provided in the
`groupDecompositionData <https://github.com/NatLabRockies/FuelLib/tree/main/fuelData/groupDecompositionData>`_ directory.
The following tutorial is included in the `FuelLib/tutorials <https://github.com/NatLabRockies/FuelLib/tree/main/tutorials>`_
as ``basic.py``. To begin, we will import the necessary modules and create a ``groupContribution`` object for the two component fuel "heptane-decane": 

.. code-block:: python

    import os
    import sys

    # Add the FuelLib directory to the Python path
    FUELLIB_DIR = os.path.dirname(os.path.dirname(__file__))
    sys.path.append(FUELLIB_DIR)
    import paths
    import FuelLib as fl

    # Create a fuel object for the fuel "heptane-decane"
    fuel = fl.fuel("heptane-decane")

Upon initialization, the ``fuel`` object will read the initial weight 
percentage composition and group decomposition data from the specified files. The object stores
vectors of the calculated fundamental properties at standard conditions for each component of the fuel as described in :ref:`eq-GCM-properties`. 
For example, we can display the fuel name, the components in the fuel, the initial composition, and the critical temperature for each component: 

.. code-block:: python

    # Display fuel name, components, initial composition, and critical temperature
    print(f"Fuel name: {fuel.name}")
    print(f"Fuel components: {fuel.compounds}")
    print(f"Initial composition: {fuel.Y_0}")
    print(f"Critical temperature: {fuel.Tc} K")

.. code-block:: none

    >> Fuel name: heptane-decane
    >> Fuel components: ['NC7H16', 'NC10H22']
    >> Initial composition: [0.7375 0.2625]
    >> Critical temperature: [549.85598051 623.69051582] K

Next, we can calculate any of the component- or mixture-level properties using the 
``groupContribution`` object. For example, we can calculate the saturated vapor pressure
for each component and the mixture at a given temperature:

.. code-block:: python

    # Calculate the saturated vapor pressure at 320 K
    T = 320 # K
    p_sat_i = fuel.psat(T)
    p_sat_mix = fuel.mixture_vapor_pressure(T)
    print(f"Saturated vapor pressure at {T} K: {p_sat_i} Pa")
    print(f"Mixture saturated vapor pressure at {T} K: {p_sat_mix} Pa")

.. code-block:: none

    >> Saturated vapor pressure at 320 K: [13735.84605413   673.28876023] Pa
    >> Mixture saturated vapor pressure at 320 K: 11117.84926875165 Pa

The following links provide more information on the :ref:`eq-GCM-correlations` and
the :ref:`eq-mixture-properties` that can be calculated using the ``groupContribution`` object.