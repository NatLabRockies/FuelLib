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

### Option 1: Install from PyPI (Recommended)

The easiest way to install FuelLib is via pip:

```bash
pip install fuellib
```

This will make the command-line tools available, including:
- `fl-export-pele` and `fl-export-converge` for exporting fuel properties
- `fl-plt-props` and `fl-plt-comp` for plotting fuel composition and properties
- `fl-fuels` for listing available fuels
- Temperature and unit conversion utilities

### Option 2: Development Installation (For Contributors)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/NatLabRockies/FuelLib.git
cd FuelLib
pip install -e '.[dev]'  # Install with development tools (docs, testing, formatting)
```

See the [Contributing](https://NatLabRockies.github.io/FuelLib/development.html) page for more detailed setup instructions and contribution guidelines.

### Optional: Use a Dedicated Conda Environment

For better dependency isolation, you can create a conda environment first:

```bash
conda create --name fuellib-env python
conda activate fuellib-env
pip install fuellib
```

## Library Usage
This repository includes multiple tutorials of ways to use FuelLib.  We recommend starting with the basic tutorial, [`tutorials/basic.py`](https://github.com/NatLabRockies/FuelLib/blob/main/tutorials/basic.py), which is documented at [https://natlabrockies.github.io/FuelLib/tutorials-basic.html](https://natlabrockies.github.io/FuelLib/tutorials-basic.html). The script [`tutorials/mixtureProperties.py`](https://github.com/NatLabRockies/FuelLib/blob/main/tutorials/mixtureProperties.py) calculates a given mixture's density, viscosity and vapor pressure from GC x GC data.  The results are plotted against data from NIST and [Edwards (2020)](https://apps.dtic.mil/sti/pdfs/AD1093317.pdf).

### Command-Line Tools
After installing FuelLib using one of the methods above, you have access to several command-line tools for plotting, unit conversion, and exporting fuel data. A comprehensive list is provided in the documentation at [https://natlabrockies.github.io/FuelLib/tutorials-cli.html](https://natlabrockies.github.io/FuelLib/tutorials-cli.html).

# Contributing
New contributions are always welcome! For detailed contribution guidelines, installation instructions, and development setup, see the [Contributing](https://NatLabRockies.github.io/FuelLib/development.html) page in the documentation.

Quick start:
1. Fork the main repository
2. Create a `newFeature` branch that contains your changes
3. Update the sphinx documentation in `newFeature`
4. Install development dependencies: `pip install -e '.[dev]'`
5. Format the source code files using the provided CLI command: `fl-format`
6. Run tests and build documentation locally to verify your changes
7. Open a Pull Request (PR) from `newFeature` on your fork to branch `main` FuelLib repository.

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

