# FuelLib
[![Language: C++17](https://img.shields.io/badge/language-Python-blue)](https://isocpp.org/)
[![DOI Badge](https://img.shields.io/badge/DOI-10.11578/dc.20250317.1-blue)](https://doi.org/10.11578/dc.20250317.1)

![CI](https://github.com/NatLabRockies/FuelLib/workflows/FuelLib-CI/badge.svg)
![Documentation](https://github.com/NatLabRockies/FuelLib/workflows/FuelLib-Docs/badge.svg)

# Overview
FuelLib (SWR-25-26) utilizes the tables and functions of the Group Contribution Method (GCM) as proposed by [Constantinou and Gani (1994)](https://doi.org/10.1002/aic.690401011) and [Constantinou, Gani and O'Connel (1995)](https://doi.org/10.1016/0378-3812(94)02593-P), with additional physical properties discussed in [Govindaraju & Ihme (2016)](https://doi.org/10.1016/j.ijheatmasstransfer.2016.06.079).  The code is based on Pavan B. Govindaraju's [Matlab implementation](https://github.com/gpavanb-old/GroupContribution) of the GCM, and has been expanded to include additional thermodynamic properties and mixture properties.  The fuel library contains gas chromatography (GC x GC) data for a variety of fuels ranging from simple single component fuels to complex jet fuels.  The GC x GC data for POSF jet fuels comes from [Edwards (2020)](https://apps.dtic.mil/sti/pdfs/AD1093317.pdf).  

## Citing this Work
If you use FuelLib in your research, please cite the following software record:

~~~
Montgomery, David, Appukuttan, Sreejith, Yellapantula, Shashank, Perry, Bruce, and Binswanger, Adam. FuelLib (Fuel Library) [SWR-25-26]. Computer Software. https://github.com/NatLabRockies/FuelLib. USDOE Office of Energy Efficiency and Renewable Energy (EERE), Office of Sustainable Transportation. Vehicle Technologies Office (VTO). 27 Feb. 2025. Web. doi:10.11578/dc.20250317.1.
~~~

## Python Environment
The following conda environment is required to run this code:
~~~
conda create --name fuellib-env matplotlib pandas scipy black 
~~~

## Running the Code
This repository includes multiple tutorials of ways to use FuelLib.  We recommend starting with the basic tutorial, `tutorials/basic.py`, which is documented at [https://NatLabRockies.github.io/FuelLib/tutorials.html#introduction]. The script `tutorials/mixtureProperties.py` calculates a given mixture's density, viscosity and vapor pressure from GC x GC data.  The results are plotted against data from NIST and [Edwards (2020)](https://apps.dtic.mil/sti/pdfs/AD1093317.pdf). 

# Contributing
New contributions are always welcome.  If you have an idea for a new feature follow these steps:
1. Fork the main repository
2. Create a `newFeature` branch that contains your changes
3. Update the sphinx documentation in `newFeature`
4. Format the source code files using the [Black code formatter](https://github.com/psf/black) by running the following command:
   ~~~
   find . -name "*.py" -print0 | xargs -0 black
   ~~~
5. Open a Pull Request (PR) from `newFeature` on your fork to branch `main` FuelLib repository.

## Sphinx Documentation
This repository uses [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) to generate documentation.  This requires the following Conda environment:
~~~
conda create --name sphinx-env sphinx sphinx_rtd_theme sphinxcontrib-bibtex pandas scipy
~~~

To view the documentation locally, build the html using the following: 
~~~
cd FuelLib/docs/
sphinx-build -M html . _build/
~~~
You should now be able to view the html by opening `FuelLib/docs/_build/html/index.html` in a web browser. 

