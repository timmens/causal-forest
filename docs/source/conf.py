#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys

import sphinx_rtd_theme


# Set variable so that todos are shown in local build
on_rtd = os.environ.get("READTHEDOCS") == "True"


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information ------------------------------------------


project = "Causal Forest"
copyright = "2020, Tim Mensinger"
author = "Tim Mensinger"

# The full version, including alpha/beta/rc tags
release = "0.0.1"


# -- General configuration ---------------------------------------------------

master_doc = "index"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx_rtd_theme",
    "nbsphinx",
]

autodoc_mock_imports = [
    "numpy",
    "pandas",
    "numba",
    "joblib",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for nbsphinx -- -------------------------------------------------
nbsphinx_execute = "never"
nbsphinx_prolog = r"""
{% set docname = 'docs/source/' + env.doc2path(env.docname, base=None) %}
.. only:: html

    .. nbinfo::
        Download the notebook :download:`here <https://nbviewer.jupyter.org/github/timmens/causal-forest/blob/master/{{ docname }}>`!
"""

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]
