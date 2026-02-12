.. _sec-exporting-to-pelephysics:

Exporting Properties for Pele
-----------------------------

The development of FuelLib was motivated by the need for more accurate liquid fuel
property prediction in computational fluid dynamics (CFD) simulations. This section desscribes
how to export properties from FuelLib for use in the spray module of the 
`PelePhysics <https://github.com/AMReX-Combustion/PelePhysics>`_ library\ :footcite:p:`owen_pelemp_2024`
for combustion simulations in the `PeleLMeX <https://github.com/AMReX-Combustion/PeleLMeX>`_ 
flow solver\ :footcite:p:`henry_de_frahan_pele_2024` \ :footcite:p:`esclapez_pelelmex_2023`.

Currently, there are two liquid property models implemented in Pele: the
`Group Contribution Method (GCM) <https://amrex-combustion.github.io/PelePhysics/Spray.html#fuellib-based-gcm>`_
and the original `PeleMP (MP) model <https://amrex-combustion.github.io/PelePhysics/Spray.html#pelemp-implementation>`_. 
The GCM is a replica of the model implemented in FuelLib and the MP model is the original model described in Owen et al. :footcite:p:`owen_pelemp_2024`

Each model has its own specific set of inputs required for Pele. The default parameters
for the export script ``Export4Pele.py`` are set for the GCM.  PelePhysics requires 
the following for each compound in the fuel for the GCM:

- Hydrocarbon family
- Molecular weight
- Critical temperature
- Critical pressure
- Critical volume
- Boiling point
- Accentric factor
- Molar volume
- Specific heat coefficients
- Latent heat of vaporization

The MP model in PelePhysics requires properties at a specified reference temperature 
for each compound in the fuel:

- Critical temperature
- Boiling point
- Latent heat of vaporization
- Specific heat
- Density
- Optionally, Antoine coefficients for vapor pressure


The export script, ``Export4Pele.py``, generates an input file containing 
the necessary properties for each compound in the fuel. The following sections 
walk through the process and the available options for exporting properties from FuelLib 
to Pele.

.. note::
    The units for PeleLMeX are MKS while the units for PeleC are CGS. This is the same for 
    the spray inputs. Therefore, when running a spray simulation coupled with PeleC, the units for the 
    liquid fuel properties must be in CGS. The default units for the ``Export4Pele.py`` script is MKS, 
    but users can specify CGS by using the ``--units cgs`` option.

Default Options
^^^^^^^^^^^^^^^
    
From the ``FuelLib`` directory, navigate to the source directory and run the following command in the terminal to export 
the required GCM parameters. The fuel here, "heptane-decane", is a binary mixture 
of heptane and decane. Note that ``--fuel_name`` is the only required input: ::
    
    cd FuelLib/source
    python Export4Pele.py --fuel_name heptane-decane


This generates the following input file, ``FuelLib/exportData/sprayPropsGCM_heptane-decane.inp``, for use in a PeleLMeX simulation: ::

    # -----------------------------------------------------------------------------
    # Liquid fuel properties for GCM in Pele
    # Fuel: heptane-decane
    # Number of compounds: 2
    # Generated: <YYY-MM-DD> <HH-MM-SS>
    # FuelLib remote URL: https://github.com/NatLabRockies/FuelLib.git
    # Git commit: <commit-hash>
    # Units: MKS
    # -----------------------------------------------------------------------------

    particles.fuel_species = NC7H16 NC10H22
    particles.Y_0 = 0.7375 0.2625
    particles.dep_fuel_species = NC7H16 NC10H22

    # Properties for NC7H16 in MKS
    particles.NC7H16_family = 0 # saturated hydrocarbons
    particles.NC7H16_molar_weight = 0.100000 # kg/mol
    particles.NC7H16_crit_temp = 549.855981 # K
    particles.NC7H16_crit_press = 2821129.514417 # Pa
    particles.NC7H16_crit_vol = 0.000425 # m^3/mol
    particles.NC7H16_boil_temp = 379.073212 # K
    particles.NC7H16_acentric_factor = 0.336945 # -
    particles.NC7H16_molar_vol = 0.000146 # m^3/mol
    particles.NC7H16_cp_a = 1636.255000 # J/kg/K
    particles.NC7H16_cp_b = 3046.511000 # J/kg/K
    particles.NC7H16_cp_c = -983.629000 # J/kg/K
    particles.NC7H16_latent = 383110.000000 # J/kg

    # Properties for NC10H22 in MKS
    particles.NC10H22_family = 0 # saturated hydrocarbons
    particles.NC10H22_molar_weight = 0.142000 # kg/mol
    particles.NC10H22_crit_temp = 623.690516 # K
    particles.NC10H22_crit_press = 2115522.932445 # Pa
    particles.NC10H22_crit_vol = 0.000592 # m^3/mol
    particles.NC10H22_boil_temp = 452.596977 # K
    particles.NC10H22_acentric_factor = 0.468050 # -
    particles.NC10H22_molar_vol = 0.000196 # m^3/mol
    particles.NC10H22_cp_a = 1630.488028 # J/kg/K
    particles.NC10H22_cp_b = 3098.105634 # J/kg/K
    particles.NC10H22_cp_c = -1024.456338 # J/kg/K
    particles.NC10H22_latent = 368035.211268 # J/kg

To include these parameters in your Pele simulation, copy the ``sprayPropsGCM_heptane-decane.inp`` 
file to the specific case directory and include the following line in your Pele input file: ::

    FILE = sprayPropsGCM_heptane-decane.inp

Additional Options
^^^^^^^^^^^^^^^^^^

There are many additional options that can be specified when running the export script:

- ``--fuel_decomp_name``: Name of the decomposition file (optional). If not provided, defaults to ``fuel_name``.
- ``--fuel_data_dir``: Specify the directory containing the fuel data files. The default is ``"FuelLib/fuelData"``.
- ``--units``: Specify the units for the properties. The default is "mks" but users can set the units to "cgs" for use in PeleC.
- ``--dep_fuel_names``: Specify which gas-phase species the liquid fuel deposits. The default is the same as the fuel name, but users can specify a single gas-phase species or a list of gas-phase species.
- ``--use_pp_keys``: Use the PelePhysics key for each compound (True or False). The default is True if keys are available.
- ``--export_dir``: Specify the directory to export the file. The default is "FuelLib/exportData".
- ``--export_mix``: Export the fuel as a single mixture species. The default is 0 or False.
- ``--export_mix_name``: Specify the name of the mixture species if ``--export_mix`` is set to True. The default is the same as the fuel name.
- ``--fuel_data_dir``: Specify the directory containing the fuel data files. The default is "FuelLib/fuelData".
- ``--liq_prop_model``: Specify the liquid property model to use. The default is ``"gcm"`` but users can set it to ``"mp"`` to export properties for the MP model in Pele.
- ``--psat_antoine``: Option to use Antoine coefficients for vapor pressure in the MP model. The default is True, but users can set it to False to not use Antoine coefficients.

Liquid Species Deposit to Single Gas-Phase Species
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To specify all liquid fuel species deposity to a single gas-phase species, run the following command: ::

    cd FuelLib/source
    python Export4Pele.py --fuel_name heptane-decane --dep_fuel_names SINGLE_GAS

This will result in the following: ::

    particles.fuel_species = NC7H16 NC10H22
    particles.Y_0 = 0.7375 0.2625
    particles.dep_fuel_names = SINGLE_GAS SINGLE_GAS

    # Properties for NC7H16 in MKS
    ...

Liquid Species Deposit to Specific Gas-Phase Species
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Alternatively, users can specify a list of gas-phase species, run the following command: ::

    python Export4Pele.py --fuel_name heptane-decane --dep_fuel_names GAS_1 GAS_2

which produces: ::

    particles.fuel_species = NC7H16 NC10H22
    particles.Y_0 = 0.7375 0.2625
    particles.dep_fuel_names = GAS_1 GAS_2

    # Properties for NC7H16 in MKS
    ...

Export Liquid Fuel as Single Mixture Species
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To export mixture properties of a multicomponent fuel as a single component, run the following command: ::

    cd FuelLib/source
    python Export4Pele.py --fuel_name heptane-decane --export_mix True

This generates the following input file, ``FuelLib/exportData/sprayPropsGCM_mixture_heptane-decane.inp``: ::

    # -----------------------------------------------------------------------------
    # Liquid fuel properties for GCM in Pele
    # Fuel: posf10264
    # Number of compounds: 1
    # Generated: <YYY-MM-DD> <HH-MM-SS>
    # FuelLib remote URL: https://github.com/NatLabRockies/FuelLib.git
    # Git commit: <commit-hash>
    # Units: MKS
    # -----------------------------------------------------------------------------

    particles.fuel_species = LIQ-MIX
    particles.Y_0 = 1.0
    particles.dep_fuel_species = GAS

    # Properties for LIQ-MIX in MKS
    particles.LIQ-MIX_family = 0 # saturated hydrocarbons
    particles.LIQ-MIX_molar_weight = 0.108418 # kg/mol
    particles.LIQ-MIX_crit_temp = 564.653893 # K
    particles.LIQ-MIX_crit_press = 2679711.894438 # Pa
    particles.LIQ-MIX_crit_vol = 0.000458 # m^3/mol
    particles.LIQ-MIX_boil_temp = 393.808839 # K
    particles.LIQ-MIX_acentric_factor = 0.363221 # -
    particles.LIQ-MIX_molar_vol = 0.000156 # m^3/mol
    particles.LIQ-MIX_cp_a = 1635.099184 # J/kg/K
    particles.LIQ-MIX_cp_b = 3056.851593 # J/kg/K
    particles.LIQ-MIX_cp_c = -991.811612 # J/kg/K
    particles.LIQ-MIX_latent = 380088.711936 # J/kg

This feature was used to generate the mixture properties for the conventional JP-8 jet fuel (posf10264),
in the `validation section <https://amrex-combustion.github.io/PelePhysics/Spray.html#spray-validation>`_ of the PelePhysics documentation,
where the liquid fuel is modeled as a single component that deposits to the HyChem gas-phase species for POSF10264.

Exporting Properties for the MP Model in Pele
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Users can export properties for the MP model in Pele by specifying ``--liq_prop_model mp`` when running the export script: ::
    
    cd FuelLib/source
    python Export4Pele.py --fuel_name heptane-decane --liq_prop_model mp

This generates the following input file, ``FuelLib/exportData/sprayPropsMP_heptane-decane.inp``, for use in a PeleLMeX simulation: ::
    
    # -----------------------------------------------------------------------------
    # Liquid fuel properties for MP in Pele
    # Fuel: posf10264
    # Number of compounds: 2
    # Generated: <YYY-MM-DD> <HH-MM-SS>
    # FuelLib remote URL: https://github.com/NatLabRockies/FuelLib.git
    # Git commit: <commit-hash>
    # Units: MKS
    # -----------------------------------------------------------------------------

    particles.fuel_species = NC7H16 NC10H22
    particles.Y_0 = 0.7375 0.2625
    particles.dep_fuel_species = NC7H16 NC10H22

    # Properties for NC7H16 in MKS
    particles.NC7H16_crit_temp = 549.855981 # K
    particles.NC7H16_boil_temp = 379.073212 # K
    particles.NC7H16_latent = 383110.000000 # J/kg
    particles.NC7H16_cp = 1636.255000 # J/kg/K
    particles.NC7H16_rho = 683.355277 # kg/m^3
    particles.NC7H16_psat = 4.1644940008887215 1351.7047368174296 -51.094643469126446 100000.0 # Pa

    # Properties for NC10H22 in MKS
    particles.NC10H22_crit_temp = 623.690516 # K
    particles.NC10H22_boil_temp = 452.596977 # K
    particles.NC10H22_latent = 368035.211268 # J/kg
    particles.NC10H22_cp = 1630.488028 # J/kg/K
    particles.NC10H22_rho = 726.195341 # kg/m^3
    particles.NC10H22_psat = 4.380101435197679 1702.1569216938776 -60.0774808903445 100000.0 # Pa

Users can choose to not use Antoine coefficients for vapor pressure in the MP model by specifying ``--psat_antoine False`` when running the export script: ::
    
    cd FuelLib/source
    python Export4Pele.py --fuel_name heptane-decane --liq_prop_model mp --psat_antoine False

This generates a similar input file as above, but without the Antoine coefficients for vapor pressure.

.. footbibliography::