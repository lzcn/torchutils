# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "torchutils"
copyright = "2025, Zhi Lu"
author = "Zhi Lu"
release = "0.0.2"


# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosummary",
]

# Generate autosummary pages
autosummary_generate = True

source_suffix = ".rst"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".AppleDouble"]

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_static_path = ["_static"]
html_title = "torchutils"
html_show_sourcelink = True

html_theme_options = {
    "description": "Essential PyTorch utilities",
}

# -- Options for autodoc ----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
}

autodoc_typehints = "description"
add_module_names = False

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

# -- Todo extension ----------------------------------------------------------

todo_include_todos = True
