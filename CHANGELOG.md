# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
The [`keepachangelog`](https://pypi.org/project/keepachangelog/) package is a dependency
used to parse and validate this file's entries against that format.

## [Unreleased]

### Added
- Pixi task automation (`fmt`, `lint`, `types`, `imports`, `test`, `pre-commit`, `docs-build`, `docs-clean`) so common dev workflows run via `pixi run <task>`.
- New dev dependencies: `ruff`, `ty`, `pytest-cov`, `lefthook`, `import-linter`, and `uv` for a faster local pip/venv workflow.
- `keepachangelog` dependency for maintaining this `CHANGELOG.md` in the Keep a Changelog format.
- Lefthook pre-commit suite (`lefthook.yaml`) running `fmt` → `lint` → `types` → `test` → `check-clean` on commit. The `import-linter` check is not yet wired into pre-commit since the layering contract (`fuellib.fuel` / `fuellib.gcm` / `fuellib.comp`) will fail broadly until the codebase is reorganized to match it; run it manually via `pixi run imports` in the meantime.
- Coverage reporting via `pytest-cov`, with a temporary `fail_under = 20` threshold, to be raised as test coverage improves.
- `fuellib.units` module exposing a shared `pint.UnitRegistry` instance (`PintUnits`) used throughout the codebase for unit-aware quantities.
- `fuellib.types` module with reusable type aliases (`FloatArray`, `PintScalar`, `PintArray`) for annotating NumPy/pint-based signatures.
- `fuellib.plot` module containing `plot_composition` and `plot_mixture_properties`, extracted from `fuellib/cli/plotting.py` so plotting logic can be imported directly from the library.
- `pint` dependency for dimensional unit handling; new `Fuel` class constants `constants.T_STP`, `constants.Sigma_g`, `constants.EpsilonByKB_g`, and `constants.MW_g` for standard-condition and ambient-gas Lennard-Jones defaults.
- `.github/copilot-instructions.md` for providing context to LLM tools.
- `docs/units.rst` with a brief example of FuelLib's pint-based unit-aware API and a link to the pint documentation for further reading; linked from the docs table of contents in `docs/index.rst`.

### Changed
- Replaced Black with Ruff + ty: `ruff format`/`ruff check` now handle formatting and linting, and `ty check` handles static type checking; `fl-format` now shells out to `ruff format`.
- Bumped `requires-python` to `>=3.12,<3.14` (from `>=3.8`); CI now runs on Python 3.12.
- CI's `Formatting` job (previously `psf/black`) now runs `ruff format --check`, `ruff check`, and `ty check`.
- **BREAKING**: `fuellib.fuel.fuel` -> `fuellib.fuel.Fuel` to conform with [PEP 8 naming conventions]("https://peps.python.org/pep-0008/#class-names") and prevent the `fuel` (module) vs. `fuel` (class) namespace clash.
- **BREAKING**: `Fuel` properties (e.g. `MW`, `Tc`, `Pc`, `Vc`, `Tb`, `Tm`, `Hf`, `Gf`, `Hv_stp`, `Lv_stp`, `Cp_stp`, `Vm_stp`, `omega`, `sigma`, `epsilonByKB`) and `utility.mixing_rule` now return/accept `pint.Quantity` values instead of raw `float`/`np.ndarray`, making units explicit and convertible via `.to(...)`.
- Refactored `fuellib/exporters/converge.py` and `fuellib/exporters/pele.py` to build exported quantities from `pint.Quantity` values instead of the removed `UnitConverter` class, adding private helpers (`_get_label`, `_get_allowed_temperature`, `_git_info`, `_filename`, `_quantity_cells`) for unit-aware formatting.
- Updated `tests/` (`test_accuracy.py`, `test_api.py`, `test_hc_identification.py`, `get_pred_and_data.py`) and `tutorials/` (`basic.py`, `compositionPlots.py`, `customFuel.py`, `hefaBlends.py`) to use the pint-based `Fuel` API (`.magnitude`, `.to(...)`, `.units`).
- Moved `keepachangelog` from core to `dev` dependencies; added `pint` as a core dependency in `pyproject.toml`.
- `.gitignore` now excludes `tests/scratch.py`.
- Updated `docs/development.rst` with a "Working with Units (pint)" section and a "Type Checking (ty)" section describing the project's type-hint expectations.
- Updated `docs/tutorials-basic.rst` and `docs/tutorials-custom-fuels.rst` example code/output for the `fuellib.fuel.fuel` -> `fuellib.fuel.Fuel` rename and the pint-based `Fuel` API.
- Updated `docs/tutorials-cli.rst` to note `fl-format` now formats with Ruff instead of Black.
- Updated `docs/tutorials-export-converge.rst` to document the new `-tu`/`--temp_units` option on `fl-export-converge`.
- Updated `docs/index.rst` package requirements to include `pint` and corrected the dev-tooling mention from Black to Ruff/ty.

### Removed
- Removed Black as a dev dependency.
- Removed the `UnitConverter` class from `fuellib/exporters/converge.py` and `fuellib/exporters/pele.py`, superseded by `pint.Quantity`-based unit handling.

### Fixed
- Fixed 15 broken `automethod` directives in `docs/fuelprops.rst` (e.g. `fuellib.fuel.viscosity_kinematic` -> `fuellib.fuel.Fuel.viscosity_kinematic`) that failed to resolve after the `fuellib.fuel.fuel` -> `fuellib.fuel.Fuel` rename.
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
