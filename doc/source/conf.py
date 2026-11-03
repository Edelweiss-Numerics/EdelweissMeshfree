# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EdelweissMeshfree"
copyright = "2024, Matthias Neuner"
authors = ["Matthias Neuner", "Thomas Mader"]
release = "v26.11"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration


# sys.path.insert(0, os.path.abspath("../../"))

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.coverage",
    "sphinx.ext.ifconfig",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
    "numpydoc",
    "sphinxcontrib.bibtex",
]

bibtex_bibfiles = ["publications.bib"]
bibtex_default_style = "unsrt"

templates_path = ["_templates"]
exclude_patterns = []

autosummary_generate = True
autoclass_content = "class"
autodoc_member_order = "groupwise"
# autodoc_typehints = "both"
# less crowded:
autodoc_typehints = "description"

autoclass_content = "init"

napoleon_use_admonition_for_notes = True
numpydoc_show_class_members = True
numpydoc_class_members_toctree = False
numpydoc_show_inherited_class_members = True


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_logo = "./logo.png"

html_static_path = ["_static"]

html_css_files = [
    "css/custom.css",
]

# for execution python code in text

try:
    pass
except ImportError:
    pass

