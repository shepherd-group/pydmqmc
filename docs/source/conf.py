# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'pydmqmc'
copyright = '2025, William Van Benschoten, Claire Kopenhafer'
author = 'William Van Benschoten, Claire Kopenhafer'
release = 'alpha'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.autodoc',
              'sphinx.ext.intersphinx',
              'numpydoc',
              'sphinx_rtd_theme',
              'sphinxcontrib.bibtex'
              ]

autodoc_typehints = "none"
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "inherited-members": True,
    "show-inheritance": True
}

autosummary_generate = True

numpydoc_show_class_members = True
numpydoc_show_inherited_class_members = True
numpydoc_attributes_as_param_list = False
numpydoc_class_members_toctree = False

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None)
}

bibtex_bibfiles = ['references.bib']

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
