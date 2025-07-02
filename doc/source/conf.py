import sys
from pathlib import Path

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "TorchCurves"
copyright = "2025, Alex Shtoff"
author = "Alex Shtoff"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

napoleon_google_docstring = True  # parse Google style
napoleon_numpy_docstring = False  # we only need one flavour

autosummary_generate = True  # generate stub *.rst files
autosummary_imported_members = True  # list re-exported symbols
autodoc_typehints = "description"  # rely on PEP-484 annotations

templates_path = ["_templates"]
exclude_patterns = []

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "alabaster"
html_static_path = ["_static"]

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # two levels up from conf.py
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # make `import torchcurves` work
