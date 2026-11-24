Citing this work
----------------

If you use FuelLib in your research, please cite the following software record:

Montgomery, David, Appukuttan, Sreejith, Yellapantula, Shashank, Perry, Bruce, and Binswanger, Adam. FuelLib (Fuel Library) [SWR-25-26]. Computer Software. https://github.com/NatLabRockies/FuelLib. USDOE Office of Energy Efficiency and Renewable Energy (EERE), Office of Sustainable Transportation. Vehicle Technologies Office (VTO). 27 Feb. 2025. Web. `doi:10.11578/dc.20250317.1 <https://doi.org/10.11578/dc.20250317.1>`_.

Or in BibTeX format:

.. code-block:: bibtex

   @misc{fuellib_2025,
      title = {FuelLib (Fuel Library) [SWR-25-26]},
      author = {Montgomery, David and Appukuttan, Sreejith and Yellapantula, Shashank and Perry, Bruce and Binswanger, Adam},
      doi = {10.11578/dc.20250317.1},
      url = {https://doi.org/10.11578/dc.20250317.1},
      howpublished = {[Computer Software] \url{https://doi.org/10.11578/dc.20250317.1}},
      year = {2025},
      month = {feb}
   }

Installation
------------

The easiest way to install FuelLib is via pip from
`pypi.org/project/fuellib <https://pypi.org/project/fuellib>`_:

.. code-block:: bash

   pip install fuellib

After installation, the ``fl`` command-line tool is available for decomposing
SMILES strings, exporting fuel data, plotting fuel properties, and simulating
distillation. See :doc:`tutorials_cli` for usage examples and options.

For a development setup, see :doc:`contributing`.

Requirements
------------

FuelLib requires Python 3.12 to 3.14 and the following packages, which are
installed automatically by ``pip``:

- numpy ≥1.19.0
- pandas ≥1.0.0
- scipy ≥1.5.0
- pyyaml ≥5.0
- matplotlib ≥3.0
- rdkit ≥2022.3.1
- pint ≥0.26.1
- pydantic ≥2.13.5
- uncertainties ≥3.2.3

Development tools (Sphinx, Ruff, ty, pytest) are available for developers
installing from source; see :doc:`contributing`.

Basic usage
-----------

See the :doc:`Basic Usage tutorial <tutorials_basic>` for an introduction to 
creating a ``Fuel`` object and predicting fuel properties.