Contributing
============

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

Pre-commit Hooks
^^^^^^^^^^^^^^^^

This repository uses `lefthook <https://lefthook.dev>`_ to run the formatting, linting,
type-checking, and test tasks automatically before each commit. After installing the
``dev`` dependencies, enable the hooks with:

.. code-block:: bash

   lefthook install

You can also run the full pre-commit suite manually with ``lefthook run pre-commit --all-files``.
Note that the ``import-linter`` module-layering check (``lint-imports``) is not yet part of
the pre-commit suite, since the codebase doesn't fully match the target module layering yet.

Test coverage is currently enforced at a low threshold (``fail_under = 20`` in
``pyproject.toml``) while the codebase is migrated; this will be raised over time.

Optional: uv
^^^^^^^^^^^^

`uv <https://docs.astral.sh/uv/>`_ is included as a dev dependency and offers a much
faster alternative to ``pip`` for installing dependencies and managing a virtual
environment:

.. code-block:: bash

   uv venv
   uv pip install -e '.[dev]'

Run ``uv lock`` to regenerate ``uv.lock`` after changing dependencies in
``pyproject.toml``, keeping installs reproducible across machines.

Developer Tasks
---------------

Common developer tasks are defined with `taskipy <https://github.com/taskipy/taskipy>`_
in the ``[tool.taskipy.tasks]`` table of ``pyproject.toml``, and can be run with
``task <name>`` (or ``uv run task <name>`` if using uv):

.. list-table::
   :header-rows: 1

   * - Task
     - Command
     - Description
   * - ``format``
     - ``ruff format``
     - Format the codebase
   * - ``lint``
     - ``ruff check --fix``
     - Lint the codebase, auto-fixing issues where possible
   * - ``types``
     - ``ty check``
     - Run static type checking
   * - ``test``
     - ``pytest``
     - Run the test suite
   * - ``pre-commit``
     - ``lefthook run pre-commit``
     - Run the full pre-commit suite (format, lint, types, test)
   * - ``build-docs``
     - ``sphinx-build -M html docs/ docs/_build``
     - Build the documentation with Sphinx
   * - ``view-docs``
     - ``bash fuellib/cli/view-docs.sh``
     - Open the built documentation in a browser

Working with Units (pint)
^^^^^^^^^^^^^^^^^^^^^^^^^

FuelLib's public API is unit-aware: physical quantities (e.g.
:attr:`Fuel.comps.mw <fuellib.fuel.components.Components.mw>`,
:attr:`Fuel.comps.tc <fuellib.fuel.components.Components.tc>`,
:attr:`Fuel.comps.pc <fuellib.fuel.components.Components.pc>`) are `pint <https://pint.readthedocs.io>`_
``Quantity`` objects rather than bare ``float``/``np.ndarray`` values. When
contributing new code that creates or consumes physical quantities:

- Always build quantities using the shared
   :data:`fuellib.Units <fuellib.utils.types.Units>` registry (also
   available as :data:`fuellib.utils.Units <fuellib.utils.types.Units>`), for example
   ``Units.Quantity(value, "K")``. Do not instantiate a new
   ``pint.UnitRegistry()``: quantities from different registries are not
   compatible with one another.
- Convert between units with ``.to("target_unit")``; access the raw numeric
  value with ``.magnitude`` only at boundaries (e.g. plotting, exporting,
  passing to non-pint-aware libraries).
- For annotations, import ``types`` from ``fuellib.utils`` and use its
   aliases: :data:`~fuellib.utils.types.Array1D` and
   :data:`~fuellib.utils.types.Array2D` for one- and two-dimensional
   ``float64`` NumPy arrays, and :data:`~fuellib.utils.types.Quantity0D`,
   :data:`~fuellib.utils.types.Quantity1D`, and
   :data:`~fuellib.utils.types.Quantity2D` for Pint quantities with scalar, one-dimensional, and
   two-dimensional magnitudes, respectively.
- Do not reintroduce bare-float return types for existing public
  :class:`~fuellib.fuel.core.Fuel`/``utility`` APIs; this would reverse an intentional breaking
  change (see ``CHANGELOG.md``).

Code Style and Naming
^^^^^^^^^^^^^^^^^^^^^

FuelLib follows `PEP 8 <https://peps.python.org/pep-0008/>`_ naming conventions
everywhere, with one deliberate exception (``greenfield.py``, below):

- Functions, methods, arguments, and variables use ``lower_snake_case``, including
  names for physical symbols (e.g. ``temp``, ``yi``, ``temp_vals``, ``num_c``,
  ``ds_fus``, ``tc``, ``rd_a``) and arrays of group contributions (e.g. ``xj``,
  ``nij``).
- Classes use ``CapWords`` and module-level constants use ``UPPER_SNAKE_CASE``
  (e.g. ``K_B``, ``T_STP``, ``MW_GAS``).
- Do not rename string keys that refer to data files (e.g. CSV column or row labels
  such as ``"Tv"``, ``"dBv"``, ``"CpAk"``, or ``"rd_A"``); these match the
  published parameter tables.
- GCM property functions are registered by their lowercased function name, so
  :meth:`predict("Tc", fuel) <fuellib.gcm.core.GCM.predict>` and
  :meth:`predict("tc", fuel) <fuellib.gcm.core.GCM.predict>` are equivalent.

Use the same name for the same quantity throughout the code base:

- Temperature: ``temp`` for a generic temperature, ``temp_vals`` for an array of
  temperatures, and ``temp_ref`` for a reference temperature.
- Named temperatures and critical properties: ``tb`` (boiling), ``tc``
  (critical), ``tm`` (melting), ``tr`` (reduced), ``tbr`` (reduced boiling),
  ``t_freeze``, ``t_flash``, ``pc``, ``vc``, and ``omega`` (acentric factor).
- Composition: ``x`` or ``xi`` (mole fractions), ``yi`` (mass fractions), ``vi``
  (volume fractions), and ``mw`` (molecular weight).
- Suffixes: ``_i`` for per-component values, ``_j`` for per-group values,
  ``_ij`` for component-group or component-component values, ``_mix`` for
  mixture values, and ``_mag`` for the magnitude of a Pint quantity (e.g.
  ``temp_mag = temp.to("K").magnitude``).
- Do not put units in names (e.g. ``temp_c`` or ``pc_bar``). Convert the quantity
  in place and note the unit in a comment, e.g.
  ``pc = fuel.comps.pc.to("bar").magnitude  # bar``.

**Exception: the Greenfield distillation model.**
``fuellib/simulate/distillation/greenfield.py`` keeps the symbols of the published
model equations, because it is dominated by math.

Code style is otherwise enforced by Ruff (``task format`` and ``task lint``) with a
line length of 88 characters and Google-style docstrings. The PEP 8 naming rules
(Ruff's ``N`` rules) are checked for every Python file, including ``tests/``,
``tutorials/``, ``docs/``, ``fuellib/cli``, and ``fuellib/exporters``; the only file
excluded from them is ``greenfield.py`` (see ``ruff.toml``). ``tests/test_style.py``
runs Ruff as part of the test suite to confirm this.

Updating the Changelog
^^^^^^^^^^^^^^^^^^^^^^

This project keeps ``CHANGELOG.md`` in the `Keep a Changelog <https://keepachangelog.com/en/1.1.0/>`_
format. The `keepachangelog <https://pypi.org/project/keepachangelog/>`_ package is a
dependency used to parse and validate entries against that format. When making a change,
add a bullet under the ``## [Unreleased]`` section using the appropriate category
(``Added``, ``Changed``, ``Deprecated``, ``Removed``, ``Fixed``, or ``Security``).

Contributing Guidelines
^^^^^^^^^^^^^^^^^^^^^^^

New contributions are always welcome! To contribute:

1. Fork the main repository on GitHub
2. Create a new branch for your feature: ``git checkout -b newFeature``
3. Make your changes and update documentation as needed
4. Ensure development dependencies are installed (see Development Setup above)
5. Format, lint, and test your code using the provided development tasks (e.g., ``uv run task pre-commit``).
6. Update the CHANGELOG (under `[UNRELEASED]`) with detailed descriptions of the contributions.
7. Open a Pull Request (PR) from your fork to the main FuelLib repository

Building and Viewing Documentation Locally
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build the documentation after installing with ``pip install -e '.[dev]'``:

.. code-block:: bash

   fl dev build-docs

The built documentation will be in ``docs/_build/html/``. Open ``index.html`` in your browser to view it, or run ``fl dev view-docs``.

To clean the build artifacts:

.. code-block:: bash

   fl dev clean-docs


Releasing
^^^^^^^^^

Releases are maintainer-gated. A maintainer first opens a release
pull request that manually sets a new PEP 440 version in ``[project].version``
in ``pyproject.toml`` and moves all release entries from ``## [Unreleased]`` into
a dated ``## [<version>] - YYYY-MM-DD`` section. Review and merge this pull
request only after the normal CI checks pass. The ``[Unreleased]`` section must
be empty when the release workflow runs.

An authorized maintainer then opens the repository's **Actions** tab, selects the
``Release`` workflow, chooses the protected ``main`` branch, and dispatches it. The
workflow validates the version and changelog, reruns the test suite, builds and
checks the source distribution and wheel, verifies that the wheel installs, and
publishes the tested artifacts. After a successful PyPI upload it creates the
``v<version>`` tag and GitHub release.

Release security setup (repository administrator)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure PyPI `trusted publishing <https://docs.pypi.org/trusted-publishers/>`_
for the ``fuellib`` project with this repository, the ``release.yml`` workflow, and
the ``pypi`` environment. *Do not create or store a PyPI API token in repository
secrets; the workflow uses a short-lived OpenID Connect token instead*.

In GitHub, create a ``pypi`` environment and require approval from one or more
release maintainers. Keep ``main`` protected: require pull requests and successful
CI checks, restrict who can push, and restrict who can run or approve deployments.
The workflow only runs its release jobs when dispatched from ``main``; its PyPI
credential is scoped to the protected environment, and its GitHub write permission
is scoped to the final tag-and-release job.