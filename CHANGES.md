This PR makes several improvements to the FuelLib project's installation, documentation, and CI workflows. The main changes include updating installation instructions and documentation to reflect new CLI tools and development workflows, switching CI and documentation builds to use editable/development installs, and modernizing the command-line exporter interface. Additionally, a new publishing guide has been added, and Sphinx documentation references and paths have been updated for consistency.

## Major Changes

### Module Organization & Architecture
- **Split monolithic `FuelLib.py` into modules:**
  - `constants.py` - Physical constants (k_B, N_A)
  - `convert.py` - Temperature and unit conversions 
  - `utility.py` - Mixture properties and droplet calculations 
  - `fuel.py` - Main Fuel class for GCM calculations
- **Reorganized repository structure for proper Python packaging:**
  - Renamed `source` -> `fuellib` and added a `pyproject.toml` for distribution via pip and conda with proper entry point configuration 
  - Created `fuellib/cli/` subpackage containing all command-line tools
  - Improved package discovery and installation with `pip install -e .`, `pip install -e '.[dev]'`, and `pip install fuellib`
  - Reworked the antiquated paths structure for managing paths between data and scripts. Users only need to `import fuellib as fl` provided a `pip install`. 

### CLI Tools Expansion & Organization
**New CLI Commands:**
- `fl-C2K`, `fl-K2C` - Celsius/Kelvin conversion utilities
- `fl-C2F`, `fl-F2C`, `fl-F2K`, `fl-K2F` - Additional temperature conversions
- `fl-eps2K` - Lennard-Jones epsilon to characteristic temperature conversion
- `fl-export-converge` - Export mixture properties for Converge CFD simulations
- `fl-export-pele` - Export properties for PelePhysics simulations
- `fl-plt-comp` - Composition plotting
- `fl-plt-props` - Properties plotting
- `fl-fuels` - List available fuels with metadata support

### Testing Improvements
- **New `test_exporters.py`:** Comprehensive integration tests for export commands (7 tests)
- **Updated `test_source_docstrings.py`:** Now validates docstrings for all public API modules
- `test_utilities.py` - Unit tests for utility functions and CLI temperature conversion commands
- `test_hc_identification.py` - Unit tests for hydrocarbon classification logic
- Simplified CI exporter job from 8 individual steps to single `test_exporters.py` call
- Total test suite: 40 tests, all passing

### Documentation & Bug Fixes
- Fixed CSV file path in `fuelprops.rst`: `../../fuelData/` → `../fuellib/data/fuelData/`
- Updated `sourcecode.rst` to reflect new file organization
- Added Sphinx docstring comments to `constants.py` and `fuel.py` attributes
- Fixed GitHub Actions failures related to decomposition metadata
- Fixed error handling for Jet A and cycloaromatic compounds

### CI/CD Modernization
- Switched to editable/development installs (`pip install -e .` and `pip install -e '.[dev]'`)
- Updated GitHub Actions workflows to use new installation methods

## Breaking Changes (v3.0.0)

Functions moved from `fuellib` namespace to submodules:

- **Temperature conversions:** `fl.C2K()` → `fl.convert.C2K()`
- **Utility functions:** `fl.mixing_rule()` → `fl.utility.mixing_rule()`  
- **Constants:** `fl.k_B` still works, but `fl.constants.k_B` recommended