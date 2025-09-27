import sys
from pathlib import Path

# Make sure we add the source code root to the path
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # two levels up from conf.py
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # make `import torchcurves` work

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
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "myst_parser",
    "sphinx_autodoc_typehints",
    "nbsphinx",
    "sphinx_copybutton",
    "sphinxext.opengraph",
]


napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_typehints = "description"  # rely on PEP-484 annotations

templates_path = ["_templates"]
exclude_patterns = []
pygments_style = "sphinx"
# mathjax_path = "https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.0.0/es5/latest?tex-mml-chtml.js"
# mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@2/MathJax.js?config=TeX-AMS_CHTML"
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"


language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
