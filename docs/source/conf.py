# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'MTH2210'
copyright = '2026, Pierre-Yves Bouchet'
author = 'Pierre-Yves Bouchet'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'myst_parser',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'matplotlib.sphinxext.plot_directive',
]



autosummary_generate = True

# Enable dollar sign ($ and $$) math syntax in MyST
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]


master_doc = 'accueil'

templates_path = ['_templates']
exclude_patterns = []

language = 'fr'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_static_path = ['_static']
html_css_files = ['custom.css']

# Remove the icons surrounding the plots
plot_html_show_formats = False
plot_html_show_source_link = False


# html_sidebars = {
#     '**': [
#         "sidebar-collapse", 
#         "sidebar-nav-bs",
#         'globaltoc.html',
#     ]
# }

