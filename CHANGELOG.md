# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The [`keepachangelog`](https://pypi.org/project/keepachangelog/) package is a dependency
used to parse and validate this file's entries against that format.

## [Unreleased]

### Added
- `TypeAlias` to type aliases in `types` module.
- `@overload` decorators to `utility.mixing_rule` to address `ty` errors.
- Sphinx-parsable type annotations and docstrings to `Fuel` attributes.
- Templates defining `class` and `module` documentation behavior.
  - `class` template separates attributes from methods and provides tables with links to each definition.
  - `module` template provides a table of all methods in the module & links to each definition.
- Docstrings to all public modules.

### Fixed
- Converted docstrings in `fuel`, `convert`, and `utility` modules to `Google` style (from `sphinx`) to enable `ruff` formatting and checking.
  - `docs/conf.py` updated to parse `Google` style docstrings.
- Ran `ruff check . --fix` to auto-format simple violations (e.g., dictionary formatting, docstring empty lines, ...).

### Changed
- `Array` aliases no longer specify dtype to lax type checking and reduce size of type annotations in documents.
- `ruff.toml` ignores `docs`, `cli`, `exporters`, `_data_locator`, and `tutorials`.
- `fuel.get_row()` -> `fuel._get_row()` to avoid documenting inner utility functions.

### Removed
- Redundant type hints in docstrings.
- Redundant attribute definitions on the `Fuel` class.
- `test_source_docstrings` in favor of `ruff` enforcing docstring rules.


## [3.0.4] - 2026-09-24

### Added
- Function string type hints to definitions in the `fuel` and `convert` modules.
  - `@overload` decorators on `convert` functions to ensure the proper types are tracked.
- `utils/` module exporting `types` and `Units` to organize FuelLib utilities.
- `ruff.toml` to thoroughly define `ruff` behavior.
- `Units` class wrapping quantity-providing dependencies, such as `pint` or `unxt`, based on their availability.
- `sphinx-autodoc-typehints` to eliminate redundancy between function signatures and docstrings (not yet implemented).

### Changed
- Moved `Units` to the `types` module to facilitate future optional dependencies.

### Fixed
- `test_api.py` now takes a flexible approach to ensuring the user interface remains consistent across versions without enforcing overly strict rules.

## [3.0.4] - 2026-09-24

### Added
- Function string type hints to definitions in the `fuel` and `convert` modules.
  - `@overload` decorators on `convert` functions to ensure the proper types are tracked.
- `utils/` module exporting `types` and `Units` to organize FuelLib utilities.
- `ruff.toml` to thoroughly define `ruff` behavior.
- `Units` class wrapping quantity-providing dependencies, such as `pint` or `unxt`, based on their availability.
- `sphinx-autodoc-typehints` to eliminate redundancy between function signatures and docstrings (not yet implemented).

### Changed
- Moved `Units` to the `types` module to facilitate future optional dependencies.

### Fixed
- `test_api.py` now takes a flexible approach to ensuring the user interface remains consistent across versions without enforcing overly strict rules.

## [3.0.3] - 2026-09-23

### Added
- Pint-backed unit support through the public `fuellib.Units` registry.
- `PintScalar` and `PintArray` type aliases for unit-aware property values.
- Pixi task automation (`fmt`, `lint`, `types`, `imports`, `test`, `pre-commit`, `docs-build`, `docs-clean`) so common dev workflows run via `pixi run <task>`.
- New dev dependencies: `ruff`, `ty`, `pytest-cov`, `lefthook`, `import-linter`, and `uv` for a faster local pip/venv workflow.
- `keepachangelog` dependency for maintaining this `CHANGELOG.md` in the Keep a Changelog format.
- Lefthook pre-commit suite (`lefthook.yaml`) running `fmt` → `lint` → `types` → `test` → `check-clean` on commit. The `import-linter` check is not yet wired into pre-commit since the layering contract (`fuellib.fuel` / `fuellib.gcm` / `fuellib.comp`) will fail broadly until the codebase is reorganized to match it; run it manually via `pixi run imports` in the meantime.
- Coverage reporting via `pytest-cov`, with a temporary `fail_under = 1` threshold, to be raised as test coverage improves.

### Changed
- **BREAKING**: Fuel properties and temperature-dependent calculations now use Pint `Quantity` values. Supply dimensional inputs with units and use `.to(unit).magnitude` only when serializing or plotting.
- Updated property plotting, exporters, tutorials, tests, and baseline data for the unit-aware API.
- Converge exports now always use MKS units; the `-u`/`--units` option was removed.
- The Converge `-m`/`--export-mix` and Pele `-m`, `-pp`, and `-psat` options are action flags that are enabled by their presence.
- Replaced Black with Ruff + ty: `ruff format`/`ruff check` now handle formatting and linting, and `ty check` handles static type checking; `fl-format` now shells out to `ruff format`.
- Bumped `requires-python` to `>=3.12,<3.14` (from `>=3.8`); CI now runs on Python 3.12.
- CI's `Formatting` job (previously `psf/black`) now runs `ruff format --check`, `ruff check`, and `ty check`.
- **BREAKING**: `fuellib.fuel.fuel` -> `fuellib.fuel.Fuel` to conform with [PEP 8 naming conventions](https://peps.python.org/pep-0008/#class-names) and prevent the `fuel` (module) vs. `fuel` (class) namespace clash.

### Removed
- Removed Black as a dev dependency.

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
