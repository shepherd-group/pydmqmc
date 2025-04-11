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

extensions = ['sphinx_automodapi.automodapi',
              'sphinx_rtd_theme',]

templates_path = ['_templates']
exclude_patterns = []

numpydoc_show_class_members = False
automodsumm_inherited_members = True
automodapi_inheritance_diagram = False

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
