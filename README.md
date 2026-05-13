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

## Installation

1. **Clone the repository:**
   ~~~
   git clone https://github.com/NatLabRockies/FuelLib.git
   cd FuelLib
   ~~~

2. **Create and activate the conda environment:**
   ~~~
   conda create --name fuellib-env python numpy pandas scipy matplotlib
   conda activate fuellib-env
   ~~~

3. **Install FuelLib:**

   **Option A: Basic installation**:
   ~~~
   pip install -e .
   ~~~

   **Option B: Full development installation** (includes documentation, testing, and formatting tools):
   ~~~
   pip install -e ".[dev]"
   ~~~

This will install FuelLib and make the command-line exporters (`fl-export-pele` and `fl-export-converge`) available in your environment. If you installed with `[dev]`, you can also use `fl-build-docs` and `fl-clean-docs` to manage documentation, and `fl-format` to format code.

## Running the Code
This repository includes multiple tutorials of ways to use FuelLib.  We recommend starting with the basic tutorial, `tutorials/basic.py`, which is documented at [https://NatLabRockies.github.io/FuelLib/tutorials.html#introduction]. The script `tutorials/mixtureProperties.py` calculates a given mixture's density, viscosity and vapor pressure from GC x GC data.  The results are plotted against data from NIST and [Edwards (2020)](https://apps.dtic.mil/sti/pdfs/AD1093317.pdf).

### Command-Line Exporters
After installing FuelLib with `pip install -e .`, you can use the command-line exporters to generate fuel properties for CFD simulations:

- `fl-export-pele`: Export fuel properties for PelePhysics simulations
- `fl-export-converge`: Export fuel properties for Converge simulations

For example:
```bash
fl-export-pele -f posf10264
fl-export-converge -f posf10325
```

Run `fl-export-pele -h` or `fl-export-converge -h` for detailed usage information. 

# Contributing
New contributions are always welcome.  If you have an idea for a new feature follow these steps:
1. Fork the main repository
2. Create a `newFeature` branch that contains your changes
3. Update the sphinx documentation in `newFeature`
4. Format the source code files using the provided CLI command:
   ~~~
   fl-format
   ~~~
   (Requires installing with `pip install -e ".[dev]"`. Uses Black version `26.3.1`.)
5. Open a Pull Request (PR) from `newFeature` on your fork to branch `main` FuelLib repository.

## Sphinx Documentation
This repository uses [Sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html) to generate documentation.

To build the documentation, first install FuelLib with development support:
~~~
pip install -e ".[dev]"
~~~

Then use the provided CLI command:
~~~
fl-build-docs
~~~

The HTML documentation will be generated in `docs/_build/html/`. Open `docs/_build/html/index.html` in your web browser to view it. 

