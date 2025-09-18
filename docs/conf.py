# Configuration file for the Sphinx documentation builder.
import os
import sys

sys.path.insert(0, os.path.abspath("../"))


# -- Project information -----------------------------------------------------

project = "torchutils"
copyright = "2025, Zhi Lu"
author = "Zhi Lu"
release = "0.0.1"


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
    "sphinx_copybutton",
]

# Generate autosummary pages
autosummary_generate = True

source_suffix = ".rst"

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", ".AppleDouble"]

# The master toctree document.
master_doc = "index"

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_title = "torchutils"
html_show_sourcelink = True

# Sphinx Book Theme options
html_theme_options = {
    "repository_url": "https://github.com/yourusername/torchutils",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": True,
    "path_to_docs": "docs",
    "repository_branch": "main",
    "home_page_in_toc": True,
    "show_toc_level": 2,
    "navigation_with_keys": True,
    "show_navbar_depth": 2,
}

# -- Options for autodoc ----------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "member-order": "bysource",
}

# Don't show class signature with the class' name.
autodoc_class_signature = "separated"

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "torch": ("https://pytorch.org/docs/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "ignite": ("https://pytorch.org/ignite/master", None),
    "pillow": ("https://pillow.readthedocs.io/en/stable", None),
}

# -- Todo extension ----------------------------------------------------------

todo_include_todos = True
