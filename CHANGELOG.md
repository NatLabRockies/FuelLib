# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The [`keepachangelog`](https://pypi.org/project/keepachangelog/) package is a dependency
used to parse and validate this file's entries against that format.

## [Unreleased]

### Changed
- `propertiesData` .csv files store Temperature vs. Value entries in individual rows rather than in a matrix format to enable non-Temperature dependent properties to be stored (e.g., "FreezePoint", "FlashPoint", ...) by setting Temp to "NaN".
  * `test_accuracy.py` and `generate_baseline.py` updated to respect this change.
  * Baseline predictions stored in `tests/baselinePredictions/mixture_baseline.csv` rather than individual per-fuel files.

### Fixed
- Added `ruff` rules `F401` and `F841` to prevent unused imports and variables; allowed `ruff` to automatically fix existing violations.

## [3.0.6] - 2026-09-30

### Added
- `fuellib.correlate` module for performing component- or mixture-wise correlations.
  * Algorithms from legacy definitions (e.g. `Fuel.density(...)`) moved to appropriate `correlation` submodules; original API maintained by wrapping new definitions.
  * `comp_idx` removed from function parameters; retained in `Fuel.{function}` for maintaining API.
  * Function names renamed to be more descriptive (no abbreviations, include equation author if available):
    * `correlate.module.fully_typed_out_equation_name{_author}`
    * `mixture_function_name` -> `mixture.function_name` (module name implicitly adds `mixture` to function call).
- Recent module additions to `importlinter` contract.
  * `lint-imports` task added to `pyproject.toml` and `lefthook.yaml`.
- Rules to `ruff.toml` enforcing code format quality:
  * *INP001*: `__init__.py` required in submodules.
  * *I001*: require organized import block.
  * *RUF022*: "dunder" (`__all__`) must be sorted.
  * *F403*: prohibit wildcard imports (`from module import *`).
  * *F405*: prohibit names possibly fetched from wildcard imports.

### Changed
- `Yi` is an optional parameter in `fuellib.correlate.mixture` functions; defaults to `fuel.Y_0`.
  * `correlate.mixture` function strings with `function(fuel, Yi, T, ...)` -> `function(fuel, T, Yi, ...)` to be consistent with function strings in `correlate.components` (`function(fuel, T, ...)`) and ensure that optional parameters are listed after non-optional.

### Fixed
- `[tool.importlinter]` now ignores imports under `if TYPE_CHECKING: ...`.

## [3.0.5] - 2026-09-29

### Added
- `fuellib.rdk` module wrapping RDKit for molecule-level structural analysis and
  property calculations: `rdk.mol` provides `from_smiles()`/`smiles()`,
  `from_inchi()`/`inchi()`, `hill_formula()`, `atom_counts()`, `has_aromatic()`,
  `has_ring()`, `has_double_bond()`, `has_branch()`, and `molecular_weight()`.
- `Fuel.smiles`, `Fuel.rdkit_mols` (cached RDKit `Mol` objects built from
  `Fuel.smiles`) for working with RDKit
- `tests/test_fuel.py` covering RDKit-backed `Fuel.nC`, `Fuel.nH`, `Fuel.MW`, and
  `Fuel.formulas` for single- and multi-compound fuels.
- `tests/test_accuracy.py::TestFuelMWAccuracy` verifying `Fuel.mean_molecular_weight`
  against expected values for `heptane` and `posf10325`.
- Excluded `.venv/` from `lefthook` pre-commit checks.


### Changed
- **BREAKING**:`Fuel.MW`, `Fuel.nC`, `Fuel.nH`, `Fuel.hc_type`, `Fuel.fam`, and
  `Fuel.formulas` are now derived from RDKit
  molecule parsing of each compound's `SMILES` string (via `fuellib.rdk.mol`)
  instead of from the Gani group-decomposition matrix (`Nij`) or the
  `gani.MW` GCM property, removing the need for hand-tuned alkyl/olefinic/aromatic group index ranges in `Fuel.__init__`.
- Added SMILES in `gcData` CSVs:
  (`decane_init.csv`, `dodecane_init.csv`, `hefa-came_init.csv`,
  `hefa-mfat_init.csv`, `hefa-tall_init.csv`, `heptane-decane_init.csv`,
  `heptane_init.csv`, `jet-a_init.csv`, `posf10264_init.csv`, `posf10289_init.csv`,
  `posf10325_init.csv`, `posf11498_init.csv`, `posf4658_init.csv`).
- Fixed `tests/baselinePredictions/generate_baseline.py` to understand recent package changes.
  - Regenerated `baselinePredictions` with new values.
- Increased minimum test coverage to 45% to prevent regression (PR coverage is at 47.28%).
  - Individual test suites in the `ci.yml` use `--no-cov` to prevent failing from module-level undercoverage.

### Removed
- `Fuel.Nij` attribute and the `gani.MW` GCM property (`fuellib.gcm.gani.MW`),
  superseded by RDKit-derived `Fuel.MW`.
- `MW` row from `fuellib/gcm/gani.csv`.

## [3.0.4] - 2026-09-28

### Added
- `gcm` module providing an extensible framework for Group Contribution Method (GCM)
  property predictions, decoupling property formulas from `Fuel`.
  - `core.PropertyProtocol`: runtime-checkable protocol defining the
    `(fuel: Fuel) -> types.Quantity1D` signature required of any GCM property function.
  - `core.GCM`: a named collection of property functions with `register_property()`,
    `list_property_fns()`, `get_property()` (case-insensitive, raises `ValueError`
    listing available properties on a miss), `predict()`, and `predict_all()`.
  - `core.GCMRegistry`: class-level registry of `GCM` instances, with `register()`,
    `list_methods()`, and case-insensitive `get_gcm()`.
  - `gcm.gani` implementing the Constantinou-Gani (and extended) method: loads
    group-contribution coefficients from `gcm.gani.csv` and registers 18 property
    functions (`Tc`, `Pc`, `Vc`, `Tb`, `Tm`, `Hf`, `Gf`, `Hv_stp`, `omega`, `Vm_stp`,
    `Cp_stp`, `Cp_B`, `Cp_C`, `rd_A`, `rd_B`, `rd_D`, `alibakhshi_phi`, `MW`) against
    a `gani` `GCMRegistry` entry, replacing formulas previously duplicated inline in
    `fuel.__init__`.
- `fuel.gani_decomp()` builds a fuel's group-decomposition matrix for the Gani method
  by reading the group decomposition file and validating that all of `fuel.compounds`
  are present (raising `ValueError` listing any missing compounds).
- Cached `fuel.gcm_properties` property that pre-computes predictions from every
  registered GCM method (via `GCMRegistry`) for a `Fuel` instance.
- `fuel.get_property(method, property_name)` for case-insensitive lookup of a GCM
  prediction from `fuel.gcm_properties`, raising `KeyError` if either the method or
  property is unregistered.
- `tests/test_gcm.py` covering the `gcm` module (`core.GCM`, `core.GCMRegistry`, and
  all `gcm.gani` property functions), including registration, error handling, and
  decomposition-matrix column resorting/partial-group-coverage behavior in
  `gani._get_decomp` (columns are reindexed to the canonical group order and any
  group missing from a fuel's decomposition is zero-filled instead of erroring).
- `docs/gcm.rst` documenting the `gcm` abstraction: `core.PropertyProtocol`,
  `core.GCM`, and `core.GCMRegistry`; how the `gani` method is registered on top
  of them; how to register new properties/methods; and the new
  `fuel.gani_decomp()`, `fuel.gcm_properties`, and `fuel.get_property()` members.
  Duplicates the GCM property table/equations from `fuelprops.rst` for
  convenience. Linked into the docs toctree (`index.rst`) and API listing
  (`sourcecode.rst`).
- `taskipy` for shortcuts to common `dev` tasks.
- Documentation for `development.rst` pointing to developer-specific tools.

### Fixed
- `ruff` now selects `E501` so the existing `line-length = 88` setting is actually
  enforced (previously configured but not checked).
- Reduced line lengths to 88 in pre-existing violations, including reflowed
  docstrings in `fuel.py`.

### Changed
- Renamed compound identifiers (not the `Compound` column header) in `decane.csv`,
  `dodecane.csv`, `hefa.csv`, `heptane.csv`, and `heptane-decane.csv` to match the
  respective gcxgc names (e.g. `NC10H22` -> `n-C10`) and dropped redundant
  annotations (e.g. `C07-Isoparaffin (lower put as same)` -> `C07-Isoparaffin`).
- `fuel.num_compounds` evaluates the length of `fuel.compounds`.
- `fuel.compounds`/`fuel.num_compounds` are now derived from `df_gcxgc["Compound"]`
  before `Nij` is read, rather than from the decomposition file's shape.

### Removed
- Unnecessary assignment of integers in `fuel.__init__` for indexing NumPy arrays when
  determining families and hydrocarbon types (e.g. `aromatics`, `branching`, `cyclos`,
  `olefins` and their `num_*` counterparts), in favor of inline slice bounds.
- Check for number of columns (`num_groups`/minimum first-order group count) in gani
  decomposition in `fuel.__init__` made unnecessary by new methods.
- Redundant computation of gani properties in `fuel.__init__` (previously computed
  twice — once for validation, once for storage — now computed once via the `gcm`
  abstraction).

## [3.0.3] - 2026-09-23

### Added
- Pint-backed unit support through the public `fuellib.Units` registry.
- `PintScalar` and `PintArray` type aliases for unit-aware property values.
- Pixi task automation (`fmt`, `lint`, `types`, `imports`, `test`, `pre-commit`, `docs-build`, `docs-clean`) so common dev workflows run via `pixi run <task>`.
- New dev dependencies: `ruff`, `ty`, `pytest-cov`, `lefthook`, `import-linter`, and `uv` for a faster local pip/venv workflow.
- `keepachangelog` dependency for maintaining this `CHANGELOG.md` in the Keep a Changelog format.
- Lefthook pre-commit suite (`lefthook.yaml`) running `fmt` → `lint` → `types` → `test` → `check-clean` on commit. The `import-linter` check is not yet wired into pre-commit since the layering contract (`fuellib.fuel` / `fuellib.gcm` / `fuellib.comp`) will fail broadly until the codebase is reorganized to match it; run it manually via `pixi run imports` in the meantime.
- Coverage reporting via `pytest-cov`, with a temporary `fail_under = 1` threshold, to be raised as test coverage improves.
- Function string type hints to definitions in the `fuel` and `convert` modules.
  - `@overload` decorators on `convert` functions to ensure the proper types are tracked.
- `utils/` module exporting `types` and `Units` to organize FuelLib utilities.
- `ruff.toml` to thoroughly define `ruff` behavior.
- `Units` class wrapping quantity-providing dependencies, such as `pint` or `unxt`, based on their availability.
- `sphinx-autodoc-typehints` to eliminate redundancy between function signatures and docstrings (not yet implemented).
- `TypeAlias` to type aliases in `types` module.
- `@overload` decorators to `utility.mixing_rule` to address `ty` errors.
- Sphinx-parsable type annotations and docstrings to `Fuel` attributes.
- Templates defining `class` and `module` documentation behavior.
  - `class` template separates attributes from methods and provides tables with links to each definition.
  - `module` template provides a table of all methods in the module & links to each definition.
- Docstrings to all public modules.

### Changed
- **BREAKING**: Fuel properties and temperature-dependent calculations now use Pint `Quantity` values. Supply dimensional inputs with units and use `.to(unit).magnitude` only when serializing or plotting.
- Updated property plotting, exporters, tutorials, tests, and baseline data for the unit-aware API.
- Converge exports now always use MKS units; the `-u`/`--units` option was removed.
- The Converge `-m`/`--export-mix` and Pele `-m`, `-pp`, and `-psat` options are action flags that are enabled by their presence.
- Replaced Black with Ruff + ty: `ruff format`/`ruff check` now handle formatting and linting, and `ty check` handles static type checking; `fl-format` now shells out to `ruff format`.
- Bumped `requires-python` to `>=3.12,<3.14` (from `>=3.8`); CI now runs on Python 3.12.
- CI's `Formatting` job (previously `psf/black`) now runs `ruff format --check`, `ruff check`, and `ty check`.
- **BREAKING**: `fuellib.fuel.fuel` -> `fuellib.fuel.Fuel` to conform with [PEP 8 naming conventions](https://peps.python.org/pep-0008/#class-names) and prevent the `fuel` (module) vs. `fuel` (class) namespace clash.
- Moved `Units` to the `types` module to facilitate future optional dependencies.
- `Array` aliases no longer specify dtype to lax type checking and reduce size of type annotations in documents.
- `ruff.toml` ignores `docs`, `cli`, `exporters`, `_data_locator`, and `tutorials`.
- `fuel.get_row()` -> `fuel._get_row()` to avoid documenting inner utility functions.

### Removed
- Removed Black as a dev dependency.
- Redundant type hints in docstrings.
- Redundant attribute definitions on the `Fuel` class.
- `test_source_docstrings` in favor of `ruff` enforcing docstring rules.

### Fixed
- Pele exports now default `particles.dep_fuel_species` to the emitted fuel-species names, including when `-pp` selects PelePhysics keys; `-dep` continues to override those names.
- Addressed 40+ Ruff linting errors across the codebase:
  - B023: Fixed lambda variable binding in test loops by capturing loop variables with default parameters (15 fixes in `tests/test_api.py`).
  - SIM102: Combined nested `if` statements using `and` operator (7 fixes across `tests/test_hc_identification.py` and `tests/test_source_docstrings.py`).
  - BLE001: Replaced overly broad `except Exception` clauses with specific exception types (4 fixes in `fuellib/__init__.py`, `fuellib/exporters/pele.py`, `tests/test_exporters.py`).
  - PLW1510: Added explicit `check=False` argument to `subprocess.run()` calls (2 fixes in `tests/test_exporters.py`, `tests/test_utilities.py`).
  - RUF059: Prefixed unused unpacked variables with underscore (2 fixes in `tests/test_utilities.py`).
  - DTZ005: Added timezone argument to `datetime.now()` call in `fuellib/exporters/pele.py`.
  - UP036: Updated outdated Python version check in `fuellib/_data_locator.py`.
  - PLC0206: Fixed dictionary iteration to use `.items()` in `tests/baselinePredictions/generate_baseline.py`.
- Addressed 10+ ty typing errors across the codebase.
- `test_api.py` now takes a flexible approach to ensuring the user interface remains consistent across versions without enforcing overly strict rules.
- Converted docstrings in `fuel`, `convert`, and `utility` modules to `Google` style (from `sphinx`) to enable `ruff` formatting and checking.
  - `docs/conf.py` updated to parse `Google` style docstrings.
- Ran `ruff check . --fix` to auto-format simple violations (e.g., dictionary formatting, docstring empty lines, ...).

## [3.0.1] - 2026-06-25

### Added
- New CLI commands: `fl-C2K`, `fl-K2C`, `fl-C2F`, `fl-F2C`, `fl-F2K`, `fl-K2F` (temperature conversions), `fl-eps2K` (Lennard-Jones epsilon to characteristic temperature), `fl-export-converge`, `fl-export-pele` (CFD export), `fl-plt-comp`, `fl-plt-props` (plotting), and `fl-fuels` (list available fuels).
- `fuellib/cli/` subpackage containing all command-line tools.
- `test_exporters.py`: integration tests for export commands.
- `test_utilities.py` and `test_hc_identification.py`: unit tests for utility functions and hydrocarbon classification logic.

### Changed
- Split monolithic `FuelLib.py` into `constants.py`, `convert.py`, `utility.py`, and `fuel.py`.
- Renamed `source` package to `fuellib` and added `pyproject.toml` for distribution via pip and conda, with proper entry point configuration.
- Switched to editable/development installs (`pip install -e .` and `pip install -e '.[dev]'`).
- Simplified CI exporter job from 8 individual steps to a single `test_exporters.py` call.
- Updated `sourcecode.rst` to reflect the new file organization.
- **Breaking:** functions moved from the `fuellib` namespace to submodules: `fl.C2K()` → `fl.convert.C2K()`, `fl.mixing_rule()` → `fl.utility.mixing_rule()`. `fl.k_B` still works, but `fl.constants.k_B` is recommended.

### Fixed
- Fixed CSV file path in `fuelprops.rst`: `../../fuelData/` → `../fuellib/data/fuelData/`.
- Fixed GitHub Actions failures related to decomposition metadata.
- Fixed error handling for Jet A and cycloaromatic compounds.

## [0.0.0] - YYYY-MM-DD

### Added
- Feature 1
- Feature 2...

### Fixed
- Fix 1
- Fix 2...

### Changed
- Change 1
- Change 2...
