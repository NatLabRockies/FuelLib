# FuelLib — Copilot Instructions

FuelLib is a Python library implementing the Group Contribution Method (GCM,
Constantinou & Gani 1994/1995, with extensions from Govindaraju & Ihme 2016)
for predicting thermodynamic and mixture properties of fuels from their
molecular composition. It ships pre-tabulated GC×GC compositional data for
many real fuels (single-component to complex jet fuels, e.g. POSF Jet A/JP-5/
JP-8) plus CLI tools for plotting, unit conversion, and exporting fuel
property tables for CFD solvers (CONVERGE, PelePhysics).

All physical quantities in the public API are unit-aware `pint.Quantity`
values (see "Units" below) — this is a recent, intentional migration; do not
"simplify" return types back to bare floats/`np.ndarray`.

## Package layout (`fuellib/`)

- `fuel.py` — core `Fuel` class. Given a fuel name (and optional custom
  decomposition/data directory), computes GCM-derived properties:
  molecular weight, critical temperature/pressure/volume, boiling/melting
  temperature, enthalpies, specific heat, acentric factor, Lennard-Jones
  parameters, etc. Most public attributes are `pint.Quantity` arrays
  (`PintArray`) indexed per-compound.
- `constants.py` — physical constants (`k_B`, `N_A`) and standard-condition /
  ambient-gas defaults (`T_STP`, `Sigma_g`, `EpsilonByKB_g`, `MW_g`), all as
  `pint.Quantity`.
- `convert.py` — temperature conversion functions (C/K/F) and related
  Lennard-Jones conversions.
- `utility.py` — mixture math, notably `mixing_rule` (arithmetic/geometric
  mean over compounds), operating on `pint.Quantity` inputs.
- `units.py` — defines the single shared `PintUnits = pint.UnitRegistry()`
  instance. Always import and use this registry rather than creating a new
  `pint.UnitRegistry()`, so quantities remain compatible across the package.
- `types.py` — shared type aliases: `FloatArray` (`np.ndarray` of floats),
  `PintScalar`, `PintArray` (pint quantities wrapping a scalar/array).
- `plot.py` — `plot_composition` and `plot_mixture_properties`, importable
  directly as library functions (also used by the `fl-plt-*` CLIs).
- `_data_locator.py` — internal helpers for locating/validating fuel data
  directories (built-in vs. user-supplied custom fuel directories).
- `data/` — packaged data:
  - `fuelData/gcData/` — GC×GC compositional data by weight percent
  - `fuelData/groupDecompositionData/` — functional group decompositions
  - `fuelData/propertiesData/` — measured/reference data used for validation
  - `fuelData/fuel_metadata.yaml` — maps fuel names → decomposition files and
    metadata (including links to experimental data for plots)
  - `gcmTableData/` — pre-tabulated group contribution coefficients
- `exporters/` — CLI exporters that build per-compound and mixture property
  tables and write CFD-solver-ready files:
  - `converge.py` (`fl-export-converge`) — CONVERGE format
  - `pele.py` (`fl-export-pele`) — PelePhysics format
  - Both use `pint.Quantity`-based helpers (`_get_label`, `_quantity_cells`,
    `_filename`, `_get_allowed_temperature`, `_git_info`) instead of a
    hand-rolled unit converter.
- `cli/` — command-line entry points (see table below):
  - `temp_converter.py`, `transport_props_converter.py` — unit conversion
  - `plotting.py` — thin CLI wrappers around `fuellib.plot`
  - `fuel_manager.py` — lists available built-in/custom fuels
  - `build_docs.py`, `clean_docs.py`, `format_code.py` — dev tooling wrappers

## CLI tools (installed as console scripts via `pyproject.toml`)

| Command | Purpose |
|---|---|
| `fl-plt-comp -f FUEL_NAME` | Plot fuel composition |
| `fl-plt-props -f FUEL_NAME [...]` | Plot properties vs. temperature |
| `fl-fuels [-dir DIR]` | List available fuels (built-in or custom) |
| `fl-export-converge -f FUEL_NAME` | Export property tables for CONVERGE |
| `fl-export-pele -f FUEL_NAME` | Export property tables for PelePhysics |
| `fl-C2K`, `fl-K2C`, `fl-C2F`, `fl-F2C`, `fl-F2K`, `fl-K2F` | Temperature unit conversions |
| `fl-eps2K` | Lennard-Jones epsilon (J/mol) → characteristic temperature (K) |
| `fl-build-docs` / `fl-clean-docs` | Build/clean Sphinx docs |
| `fl-format` | Format code (shells out to `ruff format`) |

Run any command with `-h`/`--help` for full usage; also documented in
[docs/tutorials-cli.rst](../docs/tutorials-cli.rst).

## Units (pint)

- Always get/construct quantities via `fuellib.units.PintUnits` (re-exported
  as `fuellib.PintUnits`), e.g. `PintUnits.Quantity(value, "K")`.
- Convert with `.to("target_unit")`; get raw numeric values with
  `.magnitude`; never assume a bare float/array where the API documents a
  `pint.Quantity`.
- `Fuel` (in `fuel.py`) is the renamed/PEP 8-compliant class — previously
  `fuellib.fuel.fuel` (lowercase). Import as `from fuellib import Fuel` or
  `fuellib.Fuel`.

## Tests & tutorials

- `tests/` — pytest suite: `test_accuracy.py` (CI guard that new model
  changes don't regress prediction accuracy vs. `baselinePredictions/`),
  `test_api.py` (public API/signature drift + smoke evaluations),
  `test_source_docstrings.py` (enforces numpy-style docstrings with
  `:param:`/`:type:`/`:return:`/`:rtype:`), `test_hc_identification.py`,
  `test_utilities.py`, `test_exporters.py`. `get_pred_and_data.py` is a
  shared helper for computing predictions/loading validation data.
- `tutorials/` — runnable example scripts (`basic.py`, `compositionPlots.py`,
  `customFuel.py`, `hefaBlends.py`, `mixtureProperties.py`,
  `decompose_cg.py`). These are excluded from Ruff linting but should still
  run cleanly against the current `Fuel`/pint API.

## Dev workflow (pixi tasks, see `pyproject.toml` / `lefthook.yaml`)

- `pixi run fmt` — `ruff format .`
- `pixi run lint` — `ruff check . --fix`
- `pixi run types` — `ty check`
- `pixi run imports` — `lint-imports` (import-linter; not yet wired into
  pre-commit because current layering doesn't match the target contract)
- `pixi run test` — `pytest` (includes doctest-modules and coverage)
- `pixi run pre-commit` — runs the full lefthook pre-commit suite
  (fmt → lint → types → test → check-clean)
- `pixi run docs-build` / `docs-clean` — Sphinx documentation

Without pixi: `pip install -e '.[dev]'`, then use the equivalent `ruff`,
`ty`, `pytest`, `lint-imports` commands directly, or the `uv run ...`
equivalents.

## Conventions

- Docstrings use numpy style with `:param:`/`:type:`/`:return:`/`:rtype:`
  fields (enforced by `test_source_docstrings.py` and Ruff's `pydocstyle`).
- Keep the public API (`fuellib/__init__.py` `__all__`, class/function
  signatures) stable; document any intentional breaking change in
  [CHANGELOG.md](../CHANGELOG.md) and update `tests/test_api.py`.
- CHANGELOG follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
  format (validated via the `keepachangelog` dev dependency).
