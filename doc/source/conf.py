import os
import sys

from docutils import nodes

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "EdelweissMeshfree"
copyright = "2024, Matthias Neuner"
authors = ["Matthias Neuner", "Thomas Mader"]
release = "v25.08"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../../"))

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
]

# Mock imports for compiled Cython extensions and hard dependencies that may
# not be present in the documentation build environment.
autodoc_mock_imports = [
    "edelweissfe",
    "prettytable",
    "h5py",
    "netCDF4",
    "scipy",
    "edelweissmeshfree.cells.marmotcell.marmotcell",
    "edelweissmeshfree.cells.marmotcell.lagrangianmarmotcell",
    "edelweissmeshfree.cells.marmotcell.bsplinemarmotcell",
    "edelweissmeshfree.cellelements.marmotcellelement.marmotcellelement",
    "edelweissmeshfree.cellelements.marmotcellelement.lagrangianmarmotcellelement",
    "edelweissmeshfree.materialpoints.marmotmaterialpoint.mp",
    "edelweissmeshfree.meshfree.approximations.marmot.marmotmeshfreeapproximation",
    "edelweissmeshfree.meshfree.kernelfunctions.marmot.marmotmeshfreekernelfunction",
    "edelweissmeshfree.particles.marmot.marmotparticlewrapper",
    "edelweissmeshfree.mpmmanagers.utils",
    "edelweissmeshfree.fieldoutput.mpresultcollector",
    "edelweissmeshfree.solvers.base.parallelization",
]

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


def doi_role(name, rawtext, text, lineno, inliner, options={}, content=[]):
    # rendered = nodes.Text(text)
    uri = "http://dx.doi.org/" + text
    ref = nodes.reference(rawtext, text, refuri=uri)
    return [nodes.literal("", "", ref)], []


def setup(app):
    app.add_role("doi", doi_role)
