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
- **Code formatting & linting:** Ruff
- **Type checking:** ty
- **Testing:** pytest, pytest-cov

Optional: Conda Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use a specific conda environment:

.. code-block:: bash

   conda create --name fuellib-dev-env 
   conda activate fuellib-dev-env
   conda install -c conda-forge rust
   pip install -e '.[dev]'

Optional: Pixi Environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have `pixi <https://pixi.sh>`_ installed, you can use the ``dev`` environment
and its tasks instead of managing a virtual environment by hand:

.. code-block:: bash

   pixi run -e dev fmt      # ruff format .
   pixi run -e dev lint     # ruff check . --fix
   pixi run -e dev types    # ty check
   pixi run -e dev test     # pytest (with coverage)

.. note::

   Building ``grimp`` (a dependency of ``import-linter``) requires a Rust
   toolchain. On macOS you may need to run ``brew install rust``.

Pre-commit Hooks
-----------------

This repository uses `lefthook <https://lefthook.dev>`_ to run the formatting, linting,
type-checking, and test tasks automatically before each commit. After installing the
``dev`` dependencies, enable the hooks with:

.. code-block:: bash

   lefthook install

You can also run the full pre-commit suite manually with ``pixi run -e dev pre-commit``.
Note that the ``import-linter`` module-layering check (``pixi run -e dev imports``) is not
yet part of the pre-commit suite, since the codebase doesn't fully match the target module
layering yet.

Test coverage is currently enforced at a low threshold (``fail_under = 20`` in
``pyproject.toml``) while the codebase is migrated; this will be raised over time.

Optional: uv
------------

`uv <https://docs.astral.sh/uv/>`_ is included as a dev dependency and offers a much
faster alternative to ``pip`` for installing dependencies and managing a virtual
environment:

.. code-block:: bash

   uv venv
   uv pip install -e '.[dev]'

Run ``uv lock`` to regenerate ``uv.lock`` after changing dependencies in
``pyproject.toml``, keeping installs reproducible across machines.

Updating the Changelog
-----------------------

This project keeps ``CHANGELOG.md`` in the `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_
format. The `keepachangelog <https://pypi.org/project/keepachangelog/>`_ package is a
dependency used to parse and validate entries against that format. When making a change,
add a bullet under the ``## [Unreleased]`` section using the appropriate category
(``Added``, ``Changed``, ``Deprecated``, ``Removed``, ``Fixed``, or ``Security``).

Contributing Guidelines
-----------------------

New contributions are always welcome! To contribute:

1. Fork the main repository on GitHub
2. Create a new branch for your feature: ``git checkout -b newFeature``
3. Make your changes and update documentation as needed
4. Ensure development dependencies are installed (see Development Setup above)
5. Format and lint your code using ``fl-format`` (or ``pixi run -e dev fmt``) and ``pixi run -e dev lint``

6. Run tests to verify your changes. See `.github/workflows/ci.yml` for the most up-to-date list of tests run in CI
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
