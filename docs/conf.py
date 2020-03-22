# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'LearnRL'
copyright = '2020, Mathïs Fédérico'
author = 'Mathïs Fédérico'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.

# Add autodoc and napoleon to the extensions list
extensions = ['autoapi.extension', 'sphinx.ext.autodoc', 'sphinx.ext.napoleon']
master_doc = 'index'
autoapi_dirs = ['../learnrl']

intersphinx_mapping = {
    'python': ('https://docs.python.org/', None),
    'numpy': ('http://docs.scipy.org/doc/numpy/', None)
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# -- Version control GitHub --------------------------------------------------
html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "MathisFederico", # Username
    "github_repo": "LearnRL", # Repo name
    "github_version": "dev", # Version
    "conf_py_path": "/docs/", # Path in the checkout to the docs root
}
 
