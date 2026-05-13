Exporting Properties for Converge
----------------------------------

The export script, ``fl-export-converge``, generates property data files for use in Converge CFD simulations. 
There are two options for exporting data:

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
- Mass specific heat
- Thermal conductivity

.. note::
    The property predictions of individual compounds or for the mixture may not be valid from the specified ``temp_min`` to ``temp_max``. 
    Constant values are set for temperatures below the freezing point of the mixture or above 
    the minimum critical temperature of all compounds in the fuel. These temperature values will be noted in the 
    terminal output and should be considered when using the mixture properties in a simulation.

This example walks through the process and the available options for exporting GCM-based properties for 
"posf10325", which is conventional Jet-A, using the ``fl-export-converge`` command.

Default Options
^^^^^^^^^^^^^^^
    
After installing FuelLib with ``pip install fuellib``, run the following command in the terminal, noting that ``--fuel_name`` is the only required input: ::
    
    fl-export-converge --fuel_name posf10325

Or using the short option ``-f``: ::

    fl-export-converge -f posf10325


This generates the files for each compound and a composition description in ``FuelLib/exportData/posf10325`` with  
property predictions from 0 K to 1000 K for use in a Converge simulation.

Additional Options
^^^^^^^^^^^^^^^^^^

There are several additional options that can be specified when running the ``fl-export-converge`` command:

- ``-dir, --fuel_data_dir PATH``: Directory containing the fuel data files. Default: ``FuelLib/fuelData``.
- ``-u, --units {mks,cgs}``: Units for the properties. Default: ``mks``.
- ``-t, --temp_min K``: Minimum temperature for property calculations. Default: ``0``.
- ``-T, --temp_max K``: Maximum temperature for property calculations. Default: ``1000``.
- ``-s, --temp_step K``: Step size for temperature. Default: ``10``.
- ``-o, --export_dir PATH``: Directory to export the file. Default: ``./exportData``.
- ``-m, --export_mix {true,false}``: Export mixture properties only (no individual components). Default: ``false``.

For example, run the following command to export mixture properties from 273 K to 550 K with 5 K steps: ::
    
    fl-export-converge -f posf10325 -m true -t 273 -T 550 -s 5

Or with long options: ::
    
    fl-export-converge --fuel_name posf10325 --export_mix true --temp_min 273 --temp_max 550 --temp_step 5


This generates the file ``FuelLib/exportData/mixturePropsGCM_posf10325.csv`` with mixture 
property predictions from 273 K to 550 K for use in a Converge simulation.

.. warning::
    Mixture properties for critical temperature, latent heat, and specific heat are provided by :ref:`conventional-mixing-rules` and need additional validation.

.. footbibliography::