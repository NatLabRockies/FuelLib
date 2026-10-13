# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import csv
import os
import sys

# Add parent directory to path so we can import fuellib
DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(DOCS_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)


# -- Project information -----------------------------------------------------

project = "FuelLib"
copyright = "2025 Alliance for Energy Innovation, LLC"
author = ""

# The full version, including alpha/beta/rc tags
release = "2025"

# The suffix(es) of source filenames.
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"

# The name of an image file to place at the top of the sidebar.
html_logo = "comms/FuelLibLogo_Blue.png"

# -- General configuration ---------------------------------------------------

numfig = True
extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinxcontrib.bibtex",
]

# sphinx_autodoc_typehints

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
    "no-module": True,
}

bibtex_bibfiles = ["refs.bib"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# Autodoc type hints configuration
autoclass_content = "both"
autodoc_typehints = "description"
simplify_optional_unions = (
    False  # Show Optional[Union[A, B]] instead of Union[A, B, None]
)
typehints_use_signature = True  # Show parameter types in signature
typehints_use_signature_return = True  # Show return type in signature
typehints_defaults = "braces"  # "param (type, (default: value)) -- description"

# Explicitly tell Napoleon to parse Google docstrings
napoleon_google_docstring = True
napoleon_numpy_docstring = False  # Turn off NumPy if you only use Google
napoleon_include_init_with_doc = True


# -- Generated tables --------------------------------------------------------

REF_COMPOUND_COLUMNS = ["Family", "Carbon Number", "Reference Compound", "SMILES"]


def write_ref_compounds_table(app):
    """Write a trimmed copy of refCompounds.csv for display in the docs."""
    source = os.path.join(PROJECT_ROOT, "fuellib", "data", "refCompounds.csv")
    out_dir = os.path.join(DOCS_DIR, "generated")
    os.makedirs(out_dir, exist_ok=True)
    with (
        open(source, newline="", encoding="utf-8-sig") as f_in,
        open(
            os.path.join(out_dir, "refCompounds_table.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as f_out,
    ):
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out)
        writer.writerow(REF_COMPOUND_COLUMNS)
        for row in reader:
            writer.writerow([row[col] for col in REF_COMPOUND_COLUMNS])


def setup(app):
    app.connect("builder-inited", write_ref_compounds_table)


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets)
# html_static_path = ["_static"]
# html_css_files = ["custom.css"]

html_theme_options = {"collapse_navigation": False, "navigation_depth": 3}
