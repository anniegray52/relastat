# Configuration file for Sphinx documentation builder.
# See https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
sys.path.insert(0, os.path.abspath('../rela_py'))

# -- Project information -----------------------------------------------------

project = 'codebase'
author = 'Annie Gray'

# The full version, including alpha/beta/rc tags
release = '1.0'

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Options for HTML output -------------------------------------------------

html_theme = 'alabaster'

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = 'codebasedoc'

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}

latex_documents = [
    (master_doc := 'index', project, 'codebase Documentation',
     'Annie Gray', 'manual'),
]

# -- Options for manual page output ------------------------------------------

man_pages = [
    (master_doc, 'codebase', 'codebase Documentation',
     ['Annie Gray'], 1)
]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (master_doc, 'codebase', 'codebase Documentation',
     'Annie Gray', 'codebase', 'One line description of project.',
     'Miscellaneous'),
]

# -- Extension configuration -------------------------------------------------
