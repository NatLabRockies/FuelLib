Source Code
===========

This page provides an overview of the source code available at `github.com/NatLabRockies/FuelLib <https://github.com/NatLabRockies/FuelLib>`_.

.. _source-code-structure:

FuelLib File Organization
-------------------------

- **docs:** directory containing the documentation source files
- **gcmTableData:** directory that contains the pre-tabulated group contributions
- **fuellib:** main package directory containing:

    - ``fuel.py``: core :class:`~fuellib.fuel.Fuel` class for Group Contribution Method calculations
    - ``constants.py``: physical constants (Boltzmann, Avogadro)
    - ``convert.py``: temperature conversion functions and Lennard-Jones calculations
    - ``utility.py``: utility functions for mixture properties and droplet calculations
    - ``_data_locator.py``: internal module for locating and validating fuel data directories

    - **gcm**: subpackage implementing the Group Contribution Method (GCM) abstraction
        - ``core.py``: ``GCM``/``GCMRegistry`` classes for registering and evaluating property functions
        - ``gani.py``: Constantinou-Gani (and extended) property implementations registered against the ``gani`` GCM
        - ``gani.csv``: group-contribution coefficient table used by ``gani.py``
        - ``boehm.py``: Boehm (2022) fusion-entropy (``dS_fus``) implementation registered against the ``boehm`` GCM, including hydrocarbon family identification
        - ``boehm.csv``: per-family ``dS_fus`` parameters used by ``boehm.py``

    - **correlate**: subpackage with correlation functions used to compute temperature-dependent properties of components and mixtures
        - ``components.py``: correlations for individual compound properties (e.g. density, viscosity, vapor pressure, surface tension, thermal conductivity, flash point, lower heating value)
        - ``mixture.py``: correlations for mixture properties computed from component properties and mixing rules (e.g. density, viscosity, freeze point, flash point, heat of combustion, yield sooting index, derived cetane number)
        - ``helpers.py``: shared helper functions for mixing rules and mass/mole fraction conversions

    - **utils**: subpackage with shared utilities
        - ``types.py``: type aliases and the ``Units`` quantity registry
        - ``logger.py``: ``FLLogger`` (the ``fuellib`` logger), ``set_log_level``, and ANSI color codes for terminal output

    - **rdk**: subpackage with `RDKit <https://www.rdkit.org/docs/>`_-based molecular utilities
        - ``mol.py``: functions for instantiating RDKit ``Mol`` objects from SMILES/InChI strings and for computing molecular formulas, atom counts, structural checks (aromaticity, rings, fused rings, aromatic ring counts, double bonds, branching), and molecular weight

    - **data**: package containing fuel data, metadata, and reference data

        - ``references.py``: loading, validation, and lookup of reference compounds and their measured properties
        - ``refCompounds.csv``: reference compounds (``Common_Name``, ``InChI``, ``SMILES``, ``Num_C``, ``Family``)
        - ``refProperties.csv``: measured reference properties (``Common_Name``, ``Property``, ``Units``, ``Value``, ``Error``, ``Source``)
        - **fuelData:** 
            - **gcData:** directory containing a collection of GCxGC compositional data by weight percentages
            - **groupDecompositionData:** directory containing a collection of functional group decompositions
            - **propertiesData:** directory containing measurement or predicted data used for validation
            - ``fuel_metadata.yaml``: YAML file that maps fuel names to their decomposition files and optional metadata fields
    
    - **exporters:** subpackage with CLI exporters for generating fuel properties
    
        - ``converge.py``: exporter for Converge CFD simulations (CLI: ``fl-export-converge``)
        - ``pele.py``: exporter for PelePhysics simulations (CLI: ``fl-export-pele``)

    - **cli:** subpackage with command-line interface tools for data conversion and analysis
    
        - ``temp_converter.py``: temperature conversion utilities
        - ``transport_props_converter.py``: transport properties conversion utilities
        - ``plotting.py``: plotting utilities for composition and properties
        - ``fuel_manager.py``: fuel manager utility
        - ``build_docs.py``: documentation builder utility
        - ``clean_docs.py``: documentation cleaner utility
        - ``format_code.py``: code formatter utility

- **tests:**  directory containing CI unit tests for FuelLib. The CI test checks if the cumulative error of property predictions of a new proposed model are less than or equal to the current model.
    
    - **baselinePredictions:** directory that contains baseline predictions and script ``generate_baseline.py`` for generating baseline predictions for CI testing.
    - ``test_accuracy.py``: unit test used in CI for verifying new model predictions preserve accuracy
    - ``test_source_docstrings.py``: documentation contract test that checks public source functions include required docstring fields (``:param:``, ``:type:``, ``:return:``, ``:rtype:``).
    - ``test_api.py``: combined API/signature and function-evaluation test that checks public fuellib module and class method signatures for unexpected API drift and runs representative FuelLib smoke evaluations.
    - ``test_cli_utilities.py``: unit test for utility functions and CLI commands including temperature conversion and transport property calculations.
    - ``test_hc_identification.py``: unit test for hydrocarbon classification logic.
    - ``test_ref_compounds.py``: unit test for the reference data (``data.references``) including validation, family classification, and population of missing values.
    - ``get_pred_and_data.py``: helper function used by ``test_accuracy.py`` and ``baselinePredictions/generate_baseline.py`` to compute predictions and load validation data.

- **tutorials:** directory containing example scripts that demonstrate how to use FuelLib

    - ``basic.py``: example script that demonstrates basic usage of FuelLib
    - ``hefaBlends.py``: example script that calculates properties of HEFA:Jet-A blends

Public API
----------

FuelLib's public API is continuously validated in CI using ``tests/test_api.py``.
This test verifies expected public module/class signatures and runs representative
FuelLib smoke evaluations to catch unintended behavior changes.

The project aims to keep the public API stable across releases. Any intentional
breaking API change should be explicitly documented in release notes and
accompanied by updates to tests and user-facing documentation.

Click on links below for the full auto-documentation of the API.

.. autosummary::
    :toctree: generated

    fuellib.fuel
    fuellib.constants
    fuellib.convert
    fuellib.utility
    fuellib.utils.types
    fuellib.utils.logger
    fuellib.gcm.core
    fuellib.gcm.gani
    fuellib.gcm.boehm
    fuellib.rdk.mol
    fuellib.data.references
    fuellib.correlate.components
    fuellib.correlate.mixture
    fuellib.correlate.helpers
