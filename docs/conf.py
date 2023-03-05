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
import django
import os
import sys

sys.path.insert(0, os.path.abspath(".."))
sys.path.insert(0, os.path.abspath(os.path.join("..", "openrem")))
from openrem.remapp.version import __version__, __short_version__

os.environ["DJANGO_SETTINGS_MODULE"] = "openrem.openremproject.settings"
django.setup()

# basepath = os.path.dirname(__file__)
# projectpath = os.path.abspath(os.path.join(basepath, "..", "openrem", "remapp"))
# exec(open(os.path.join(projectpath, "version.py")).read())
# version = __short_version__
# # The full version, including alpha/beta/rc tags.
# release = __version__

# -- Project information -----------------------------------------------------

project = "OpenREM"
copyright = "2013-2022, The Royal Marsden NHS Foundation Trust"  # pylint: disable=redefined-builtin
author = "OpenREM Contributers"

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = __short_version__
# The full version, including alpha/beta/rc tags.
release = __version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.graphviz",
    "sphinxarg.ext",
    "sphinx_issues",
    "sphinx_copybutton",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# html_theme = 'alabaster'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = [
    "_static",
    "_source",
]

html_css_files = [
    "css/custom.css",
]

# -- Other stuff -------------------------------------------------------------

# Sphinx issues
issues_uri = "https://bitbucket.org/openrem/openrem/issues/{issue}"  # pylint: disable=invalid-name
issues_pr_uri = "https://bitbucket.org/openrem/openrem/pull-requests/{pr}"  # pylint: disable=invalid-name
issues_commit_uri = "https://bitbucket.org/openrem/openrem/commits/{commit}"  # pylint: disable=invalid-name

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ["_build"]

# -- Options for sphinx-copybutton ----------------------------------------

copybutton_prompt_text = (
    r"/>>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: |[A-Z]\:[^>]*>|\(venv\) [A-Z]\:[^>]*>|PS [A-Z]:\\[^>]*> "
)
copybutton_prompt_is_regexp = True
copybutton_line_continuation_character = "\\"
