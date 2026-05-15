Source Code
===========

This page provides an overview of the source code available at `github.com/NatLabRockies/FuelLib <https://github.com/NatLabRockies/FuelLib>`_.

.. _source-code-structure:

FuelLib File Organization
-------------------------

- **docs:** directory containing the documentation source files
- **fuelData:** 
    - **gcData:** directory containing a collection of GCxGC compositional data by weight percentages
    - **groupDecompositionData:** directory containing a collection of functional group decompositions
    - **propertiesData:** directory containing measurement or predicted data for validation (see *fuelData/dataReferences.md*)
- **gcmTableData:** directory that contains the pre-tabulated group contributions
- **fuellib:** main package directory containing:

    - ``fuel.py``: core :class:`fuel` class for Group Contribution Method calculations
    - ``constants.py``: physical constants (Boltzmann, Avogadro)
    - ``conversions.py``: temperature conversion functions and Lennard-Jones calculations
    - ``utilities.py``: utility functions for mixture properties and droplet calculations
    - ``_data_locator.py``: internal module for locating and validating fuel data directories
    - **exporters:** subpackage with CLI exporters for generating fuel properties
    
        - ``converge.py``: exporter for Converge CFD simulations (CLI: ``fl-export-converge``)
        - ``pele.py``: exporter for PelePhysics simulations (CLI: ``fl-export-pele``)

    - **cli:** subpackage with command-line interface tools for data conversion and analysis
    
        - ``temp_converter.py``: temperature conversion utilities (CLI: ``fl-C2K``, ``fl-K2C``, ``fl-C2F``, ``fl-F2C``, ``fl-F2K``, ``fl-K2F``)
        - ``transport_props_converter.py``: Lennard-Jones parameter conversion (CLI: ``fl-eps2K``)
        - ``plotting.py``: plotting utilities for composition and properties (CLI: ``fl-plt-comp``, ``fl-plt-props``)
        - ``_fuel_manager.py``: fuel manager utility (CLI: ``fl-fuels``)

- **tests:**  directory containing CI unit tests for FuelLib. The CI test checks if the cumulative error of property predictions of a new proposed model are less than or equal to the current model.
    
    - **baselinePredictions:** directory that contains baseline predictions and script ``generate_baseline.py`` for generating baseline predictions for CI testing.
    - ``test_accuracy.py``: unit test used in CI for verifying new model predictions preserve accuracy
    - ``test_source_docstrings.py``: documentation contract test that checks public source functions include required docstring fields (``:param:``, ``:type:``, ``:return:``, ``:rtype:``).
    - ``test_api.py``: combined API/signature and function-evaluation test that checks public fuellib module and class method signatures for unexpected API drift and runs representative FuelLib smoke evaluations.
    - ``test_utilities.py``: unit test for utility functions and CLI commands including temperature conversion and transport property calculations.
    - ``test_hc_identification.py``: unit test for hydrocarbon classification logic.
    - ``get_pred_and_data.py``: helper function used by ``test_accuracy.py`` and ``baselinePredictions/generate_baseline.py`` to compute predictions and load validation data.

- **tutorials:** directory containing example scripts that demonstrate how to use FuelLib

    - ``basic.py``: example script that demonstrates basic usage of FuelLib
    - ``compositionPlots.py``: example script that generates composition plots for a given fuel
    - ``hefaBlends.py``: example script that calculates properties of HEFA:Jet-A blends
    - ``mixtureProperties.py``: validation script that calculates properties of single component fuels and mixture properties of multicomponent fuels.

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

    fuellib