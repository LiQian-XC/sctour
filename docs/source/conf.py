import os
import sys
sys.path.insert(0, os.path.abspath('../../'))


# -- Project information -----------------------------------------------------
from datetime import datetime
import sctour

project = 'sctour'
author = 'Qian Li'
copyright = f'{datetime.now():%Y}, {author}'
release = sctour.__version__


# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.viewcode',
    'myst_parser',
    'nbsphinx',
    'sphinx.ext.autosummary',
]
autosummary_generate = True
autodoc_member_order = 'bysource'
napoleon_include_init_with_doc = False
napoleon_numpy_docstring = True
napoleon_use_rtype = True
napoleon_use_param = True

# settings
master_doc = 'index'
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = 'sphinx'

intersphinx_mapping = dict(
        python=('https://docs.python.org/3/', None),
        numpy=('https://numpy.org/doc/stable/', None),
        pandas=('https://pandas.pydata.org/pandas-docs/stable/', None),
        anndata=('https://anndata.readthedocs.io/en/stable/', None),
        scanpy=('https://scanpy.readthedocs.io/en/stable/', None),
        scipy=('https://docs.scipy.org/doc/scipy/reference/', None),
        torch=('https://pytorch.org/docs/master/', None),
        matplotlib=('https://matplotlib.org/stable/', None),
)


# -- Options for HTML output -------------------------------------------------
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
