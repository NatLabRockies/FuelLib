Exporting GCM-Based Mixture Properties for Converge
---------------------------------------------------

The export script, ``Export4Converge.py`` generates property data files for use in Converge CFD simulations. 
There are two main options for exporting data:

1. **Mixture Properties**: This option generates a csv file named ``mixturePropsGCM_<fuel_name>.csv`` containing 
mixture property predictions for a given fuel over a specified temperature range. 

2. **Individual Component Properties**: This option generates a directory named ``<fuel_name>`` containing individual 
component property files (``<k>_<component_name>.csv``) and a composition file (``composition_<fuel_name>.csv``)for the fuel.

The properties include:

- Critical temperature
- Dynamic viscosity
- Surface tension
- Latent heat of vaporization
- Vapor pressure
- Density
- Specific heat
- Thermal conductivity

.. note::
    The property predictions of individual compounds or for the mixture may not be valid from the specified ``temp_min`` to ``temp_max``. 
    Constant values are set for temperatures below the freezing point of the mixture or above 
    the minimum critical temperature of all compounds in the fuel. These temperature values will be noted in the 
    terminal output and should be considered when using the mixture properties in a simulation.

This example walks through the process and the available options for exporting GCM-based properties for 
"posf10325", which is conventional Jet-A, using the ``Export4Converge.py`` script.

Default Options
^^^^^^^^^^^^^^^
    
From the ``FuelLib`` directory, run the following command in the terminal, noting that ``--fuel_name`` is the only required input: ::
    
    cd FuelLib/source
    python Export4Converge.py --fuel_name posf10325


This generates the files for each compound and a composition description in ``FuelLib/exportData/posf10325`` with  
property predictions from 0 K to 1000 K for use in a Converge simulation.

Additional Options
^^^^^^^^^^^^^^^^^^

There are several additional options that can be specified when running the export script:

- ``--units``: Specify the units for the mixture properties. The default is "mks" but users can set the units to "cgs".
- ``--temp_min``: Specify the minimum temperature. The default is 0 K.
- ``--temp_max``: Specify the maximum temperature. The default is 1000 K.
- ``--temp_step``: Specify the temperature step size. The default is :math:`\Delta T = 10` K.
- ``--export_dir``: Specify the directory to export the file. The default is "FuelLib/exportData".
- ``--fuel_data_dir``: Specify the directory containing the fuel data files. The default is "FuelLib/fuelData".
- ``--export_mix``: Set this flag to export mixture properties only. If not set, individual component properties and composition are exported.

For example, run the following command in the terminal: ::
    
    cd FuelLib/source
    python Export4Converge.py --fuel_name posf10325 --export_mix True --temp_min 273 --temp_max 550


This generates the file ``FuelLib/exportData/mixturePropsGCM_posf10325.csv`` with mixture 
property predictions from 273 K to 550 K for use in a Converge simulation.

.. warning::
    Mixture properties for critical temperature, latent heat, and specific heat are provided by :ref:`conventional-mixing-rules` and need additional validation.

.. footbibliography::