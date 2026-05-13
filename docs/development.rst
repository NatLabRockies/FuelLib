Contributing to FuelLib
=======================

We welcome contributions! This page covers how to set up your development environment, make changes, and submit pull requests.

Development Setup
-----------------

Clone the repository and install in editable mode with development dependencies:

.. code-block:: bash

   git clone https://github.com/NatLabRockies/FuelLib.git
   cd FuelLib
   pip install -e '.[dev]'

This installs FuelLib with all development tools:

- **Documentation:** Sphinx, sphinx-rtd-theme, sphinxcontrib-bibtex
- **Code formatting:** Black
- **Testing:** pytest
- **Plotting:** matplotlib

Optional: Conda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use a specific conda environment:

.. code-block:: bash

   conda create --name fuellib-env python numpy pandas scipy matplotlib
   conda activate fuellib-env
   pip install -e '.[dev]'

Contributing Guidelines
-----------------------

New contributions are always welcome! To contribute:

1. Fork the main repository on GitHub
2. Create a new branch for your feature: ``git checkout -b newFeature``
3. Make your changes and update documentation as needed
4. Ensure development dependencies are installed (see Development Setup above)
5. Format your code using Black:

   .. code-block:: bash

      fl-format

6. Run tests to verify your changes:

   .. code-block:: bash

      pytest tests/

7. Open a Pull Request (PR) from your fork to the main FuelLib repository

Building and Viewing Documentation Locally
-------------------------------------------

To build the documentation after installing with ``pip install -e '.[dev]'``:

.. code-block:: bash

   fl-build-docs

The built documentation will be in ``docs/_build/html/``. Open ``index.html`` in your browser to view it.

To clean the build artifacts:

.. code-block:: bash

   fl-clean-docs
