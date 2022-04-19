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
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------
from datetime import datetime
import sctour

project = 'sctour'
author = 'Qian Li'
copyright = f'{datetime.now():%Y}, {author}'

# The full version, including alpha/beta/rc tags
release = sctour.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'matplotlib.sphinxext.mathmpl',
    'matplotlib.sphinxext.only_directives',
    'matplotlib.sphinxext.plot_directive',
    'matplotlib.sphinxext.ipython_directive',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'myst_parser',
    'nbsphinx'
]

# Add any paths that contain templates here, relative to this directory.
source_suffix = ['.rst', '.md']
master_doc = 'index'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

pygments_style = 'sphinx'
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'
napoleon_numpy_docstring = True

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

intersphinx_mapping = dict(
        python=('https://docs.python.org/3/', None),
        numpy=('https://numpy.org/doc/stable/', None),
        pandas=('https://pandas.pydata.org/pandas-docs/stable/', None),
        anndata=('https://anndata.readthedocs.io/en/stable/', None),
        scanpy=('https://scanpy.readthedocs.io/en/stable/', None),
        scipy=('https://docs.scipy.org/doc/scipy/reference/', None),
        torch=('https://pytorch.org/docs/master/', None)
)
