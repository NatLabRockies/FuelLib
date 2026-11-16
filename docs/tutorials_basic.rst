Introduction and Basic Usage
----------------------------

FuelLib is a Python library that uses group contribution methods to estimate the 
thermophysical properties of multicomponent liquid fuels and their mixtures. This tutorial 
provides a basic introduction to using FuelLib, including installation, required input files, and example usage.

Download and Setup
^^^^^^^^^^^^^^^^^^

Install FuelLib using pip:

.. code-block:: bash

   pip install fuellib

The required dependencies will be installed automatically.

For information about contributing or installation for development, see the `Contributing <development.html>`_ page.

If you want to run the example scripts, you can either clone the repository or download individual tutorial files from the `tutorials <https://github.com/NatLabRockies/FuelLib/tree/main/tutorials>`_ directory on GitHub.

Required Input Files
^^^^^^^^^^^^^^^^^^^^^

FuelLib comes with a variety of built-in fuels with pre-populated input files, but you can also add your own custom fuels by providing the required input files. Each fuel requires one input file:

- ``fuellib/data/gc_data/<fuel_name>.csv``: the GCxGC composition of the fuel (must include a ``Weight %`` column and either a ``Reference Compound`` or a ``SMILES`` column; optional columns include ``GC-Bin`` and ``PelePhysics Key``)

Each row of the GCxGC file is matched to a compound in the reference data tables, which store the compound's SMILES, hydrocarbon family, and group decomposition:

- ``fuellib/data/ref_compounds.csv``: reference compounds with their family, carbon number, SMILES, and (optionally) measured properties.
- ``fuellib/data/ref_gani.csv``: the fundamental group decomposition of each reference compound (must have columns for the groups defined in `gani.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/gcm/gani.csv>`_).
- ``fuellib/data/ref_nannoolal.csv``: the Nannoolal group decomposition of each reference compound, used for the Nannoolal viscosity correlations (columns must be groups defined in `nannoolal.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/gcm/nannoolal.csv>`_).
- ``fuellib/data/ref_unifac.csv``: the UNIFAC subgroup decomposition of each reference compound, used for the UNIFAC 2.0 activity coefficients (columns must be subgroups defined in `unifac.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/gcm/unifac.csv>`_).

Rows are matched by ``Reference Compound`` name first (case-insensitive) and then by the InChI derived from ``SMILES``. A ``ValueError`` listing the unmatched rows is raised if any GCxGC compound is missing from the reference tables, so new compounds must first be added to the reference tables (see the `Adding Custom Fuels <tutorials_custom_fuels.html>`_ tutorial). Many examples can be found in the `fuellib/data <https://github.com/NatLabRockies/FuelLib/tree/main/fuellib/data>`_ directory in the repository.

Optionally, measured property data can be provided in ``properties_data/<fuel_name>.csv``. It is loaded into ``Fuel.prop_data`` and used for validation and plotting.

Decomposing Fuel Components into Fundamental Groups
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The decomposition process requires knowledge of the chemical structure of each component in the fuel. 
A starting decomposition can be generated automatically from SMILES strings with :func:`fuellib.gcm.gani.decompose` or the ``fl-decomp-smiles`` command (see :doc:`tutorials_cli`), but the results should be checked against the `Gani table`_.
Nannoolal decompositions for ``ref_nannoolal.csv`` can be generated in the same way with :func:`fuellib.gcm.nannoolal.decompose` or ``fl-decomp-smiles --method nannoolal``, and UNIFAC decompositions for ``ref_unifac.csv`` with :func:`fuellib.gcm.unifac.decompose` or ``fl-decomp-smiles --method unifac``.

.. warning::
    Automated decompositions must be verified. A known shortcoming is indene, where the non-aromatic C=C bond is categorized as a ``CH`` group.

For example, consider a multicomponent fuel consisting of heptane (C7H16), decane (C10H22), and (1,5-dimethylhexyl)cyclohexane (C14H28).

.. figure:: /figures/decomp-example.png
   :width: 300pt
   :align: center
   
   Chemical structures of heptane (top), decane (middle), and (1,5-dimethylhexyl)cyclohexane (bottom). The CH3 groups are highlighted in green, the CH2 groups in blue, and the CH group in gold. The additional second order structures of (1,5-dimethylhexyl)cyclohexane are shown in orange.

The decomposition of heptane and decane into the groups defined in the `Gani table`_ is straightforward, as both compounds consist of only CH3 and CH2 groups. 
(1,5-dimethylhexyl)cyclohexane is a branched cycloalkane with a six-membered ring and 
two methyl branches on the first and fifth carbon atoms of the ring, as shown in the figure above.
The six-membered ring and the branch containing the two CH3 groups bonded to a CH group 
are accounted for using the second order groups defined in the `Gani table`_. However, 
the remaining branch with a single CH3 group bonded to a CH2 group is not defined in the `Gani table`_.

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

.. note::
    The group decompositions in ``ref_gani.csv`` must follow the groups defined in the `Gani table`_, there are :math:`N_{g1} = 78` 
    first-order groups and :math:`N_{g2} = 43` second order groups. The second-order groups start with the 
    branching structure `(CH3)2CH`. Not all branching structures are defined in the `Gani table`_. We recommend
    starting with ``fuellib/data/ref_gani.csv`` and adapting the decompositions and compounds for your fuel. 

.. _Gani table: https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/gcm/gani.csv

Basic Usage
^^^^^^^^^^^

To demonstrate the usage of FuelLib, we will use the fuel "heptane_decane", which is a 
binary mixture of heptane and decane. The initial weight percentage composition is 73.75% 
heptane and 26.25% decane, and the group decompositions of both compounds are provided in
`ref_gani.csv <https://github.com/NatLabRockies/FuelLib/blob/main/fuellib/data/ref_gani.csv>`_.
The following tutorial is included in the `FuelLib/tutorials <https://github.com/NatLabRockies/FuelLib/tree/main/tutorials>`_
as ``basic.py``. To begin, we will import the necessary modules and create a ``fuel`` object for the two component fuel "heptane_decane": 

.. code-block:: python

    import fuellib as fl

    # Create a fuel object for the fuel "heptane_decane"
    fuel = fl.Fuel("heptane_decane")

Upon initialization, the ``fuel`` object will read the GCxGC composition and match each compound to the 
reference compound and group decomposition tables. Fundamental properties at standard conditions for each component 
of the fuel (e.g., ``fuel.comps.tc``, ``fuel.comps.pc``, ``fuel.comps.tb``) are evaluated lazily on first access and returned as 
vectors (pint quantities for unit checking), as described in :ref:`eq-GCM-properties`. When measured values for a 
compound are available in ``ref_compounds.csv`` they are used in place of the group-contribution prediction 
(see :ref:`sec-reference-properties`); pass ``use_ref_props=False`` to :class:`~fuellib.fuel.core.Fuel` to use only predictions.
For example, we can display the fuel name, the components in the fuel, the initial composition, and the critical temperature for each component: 

.. code-block:: python

    # Display fuel name, components, initial composition, and critical temperature
    print(f"Fuel name: {fuel.name}")
    print(f"Fuel components: {fuel.comps.names}")
    print(f"Initial composition: {fuel.y0:~P.4f}")
    print(f"Critical temperature: {fuel.comps.tc:~P.2f}") # ~P for "pretty" pint unit formatting

.. code-block:: none

    >> Fuel name: heptane_decane
    >> Fuel components: ['n-heptane', 'n-decane']
    >> Initial composition: [0.7375 0.2625]
    >> Critical temperature: [549.86 623.69] K

Note that the units for critical temperature are included as `fuel.comps.tc` returns a `pint.Quantity[np.ndarray]`. Next, we can calculate any of the component- or mixture-level properties using the 
``fuel`` object: component properties are available through ``fuel.comps``
(:class:`~fuellib.fuel.components.Components`) and mixture properties through ``fuel.mix``
(:class:`~fuellib.fuel.mixture.Mixture`). For example, we can calculate the vapor pressure
for each component and the mixture at a given temperature:

.. code-block:: python

    # Calculate the vapor pressure at 320 K
    T = fl.Units.Quantity(320, "K") # Temperature as a pint.Quantity
    pvap_i = fuel.comps.pvap(T)
    p_vap_mix = fuel.mix.pvap(fuel.y0, T)
    print(f"Component vapor pressure at {T:~P}: {pvap_i:~P.2f}")
    print(f"Mixture vapor pressure at {T:~P}: {p_vap_mix:~P.2f}")

.. code-block:: none

    >> Component vapor pressure at 320 K: [13735.85 673.29] Pa
    >> Mixture vapor pressure at 320 K: 11117.77 Pa

The following links provide more information on the :ref:`eq-GCM-correlations` and
the :ref:`eq-mixture-properties` that can be calculated using the ``groupContribution`` object.