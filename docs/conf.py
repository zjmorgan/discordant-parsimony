# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

import inspect
import importlib

# -- Project information -----------------------------------------------------

project = "discordant-parsimony"
copyright = ""
author = "Zachary Morgan"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.githubpages",
    "sphinx.ext.autodoc",
    "sphinx.ext.linkcode",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "numpydoc",
]

templates_path = ["_templates"]

exclude_patterns = ["_build", "source/modules.rst"]

root_doc = "index"

numpydoc_show_class_members = False

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"
html_permalinks_icon = "#"
html_show_sourcelink = False
html_copy_source = True

html_static_path = ["_static"]

html_theme_options = {
    "logo": {
        "image_light": "_static/discordant_logo.svg",
        "image_dark": "_static/discordant_logo.svg",
    },
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/zjmorgan/discordant-parsimony",
            "icon": "fab fa-github-square",
            "type": "fontawesome",
        },
    ],
    "external_links": [
        {
            "url": "https://single-crystal.ornl.gov",
            "name": "Single Crystal Diffraction",
        },
    ],
    "show_nav_level": 2,
}

# -- Extension configuration -------------------------------------------------

add_module_names = False


def linkcode_resolve(domain, info):
    baseurl = (
        "https://github.com/zjmorgan/discordant-parsimony/blob/main/src/{}.py"
    )
    if "py" not in domain:
        return None
    if not info["module"]:
        return None
    filename = info["module"].replace(".", "/")
    mod = importlib.import_module(info["module"])
    if os.path.basename(getattr(mod, "__file__", "")) == "__init__.py":
        filename += "/__init__"
    url = baseurl.format(filename)
    objname, *attrname = info["fullname"].split(".")
    obj = getattr(mod, objname, None)
    if obj is None:
        return url
    for attr in attrname:
        obj = getattr(obj, attr, None)
        if obj is None:
            return url
    try:
        lines = inspect.getsourcelines(obj)
        start, stop = lines[1], lines[1] + len(lines[0]) - 1
        return "{}#L{}-L{}".format(url, start, stop)
    except (TypeError, OSError):
        return url
