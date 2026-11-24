# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------

import ast
import csv
import importlib
import inspect
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
    "sphinx.ext.linkcode",
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
exclude_patterns = ["_build", "_includes", "Thumbs.db", ".DS_Store"]

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


# -- Links to the source code on GitHub --------------------------------------

GITHUB_SOURCE_URL = "https://github.com/NatLabRockies/FuelLib/blob/main"
PACKAGE_DIR = os.path.join(PROJECT_ROOT, "fuellib")


def _find_assignment_lines(source_file, name):
    """Return the (first, last) line of the top-level assignment of ``name``."""
    with open(source_file, encoding="utf-8") as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign):
            targets = node.targets
        elif isinstance(node, ast.AnnAssign):
            targets = [node.target]
        else:
            continue
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return node.lineno, node.end_lineno
    return None


def linkcode_resolve(domain, info):
    """Return the GitHub URL of the source of a documented Python object."""
    if domain != "py" or not info["module"]:
        return None
    try:
        module = importlib.import_module(info["module"])
        obj = module
        for part in info["fullname"].split("."):
            obj = getattr(obj, part)
    except Exception:
        return None

    source_file = lines = None
    try:
        obj = inspect.unwrap(obj)
        source_file = inspect.getsourcefile(obj)
        source_lines, first = inspect.getsourcelines(obj)
        lines = (first, first + len(source_lines) - 1)
    except (TypeError, OSError):
        source_file = None

    # Constants and type aliases have no source of their own: link to the
    # assignment in the module that defines them.
    if source_file is None or not os.path.abspath(source_file).startswith(PACKAGE_DIR):
        source_file = getattr(module, "__file__", None)
        if source_file is None or "." in info["fullname"]:
            return None
        lines = _find_assignment_lines(source_file, info["fullname"])

    relative = os.path.relpath(os.path.abspath(source_file), PROJECT_ROOT)
    url = f"{GITHUB_SOURCE_URL}/{relative.replace(os.sep, '/')}"
    if lines:
        url += f"#L{lines[0]}-L{lines[1]}"
    return url


# -- Generated tables --------------------------------------------------------

REF_COMPOUND_COLUMNS = ["Family", "Carbon Number", "Reference Compound", "SMILES"]


def write_ref_compounds_table(app):
    """Write a trimmed copy of ref_compounds.csv for display in the docs."""
    source = os.path.join(PROJECT_ROOT, "fuellib", "data", "ref_compounds.csv")
    out_dir = os.path.join(DOCS_DIR, "generated")
    os.makedirs(out_dir, exist_ok=True)
    with (
        open(source, newline="", encoding="utf-8-sig") as f_in,
        open(
            os.path.join(out_dir, "ref_compounds_table.csv"),
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
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {"collapse_navigation": False, "navigation_depth": 4}

# Do not list documented objects (classes, functions, ...) as entries in the
# sidebar tree below their own page.
toc_object_entries = False
