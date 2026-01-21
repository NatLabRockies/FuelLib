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
- **source:** directory containing the main source code files

    - ``Export4Converge.py``: script that exports mixture properties over a range of user specified temperatures for use in Converge simulations.
    - ``Export4Pele.py``: script that exports critical properties and initial mass fraction data for use in Pele simulations.
    - ``FuelLib.py``: class for enabling GCM predictions

- **tests:**  directory containing CI unit tests for FuelLib. The CI test checks if the cumulative error of property predictions of a new proposed model are less than or equal to the current model.
    
    - **baselinePredictions:** directory that contains baseline predictions
    - ``test_accuracy.py``: unit test used in CI for verifying new model predictions preserve accuracy
    - ``test_baseline.py``: generates .csv files for the baseline model predictions, which are stored in **baselinePredictions**
    - ``test_functions.py``: collection of functions used by ``test_baseline.py`` and ``test_accuracy.py``.

- **tutorials:** directory containing example scripts that demonstrate how to use FuelLib

    - ``basic.py``: example script that demonstrates basic usage of FuelLib
    - ``compositionPlots.py``: example script that generates composition plots for a given fuel
    - ``hefaBlends.py``: example script that calculates properties of HEFA:Jet-A blends
    - ``mixtureProperties.py``: validation script that calculates properties of single component fuels and mixture properties of multicomponent fuels.

- ``paths.py``: file that defines paths to various directories and files used in FuelLib

Source Code Auto-Documentation
------------------------------
Click on links below for the full auto-documentation.

.. autosummary::
    :toctree: generated

    FuelLib
    Export4Pele
    Export4Converge