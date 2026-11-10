# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import re
import sys

# -- Path setup --------------------------------------------------------------
# Make the source tree importable for autodoc.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

# Let the GPU kernels import on machines without a GPU.
os.environ.setdefault("NUMBA_ENABLE_CUDASIM", "1")


def _read_version() -> str:
    with open(os.path.join(_ROOT, "ipie", "_version.py"), encoding="utf-8") as fh:
        match = re.search(r'__version__\s*=\s*["\']([^"\']+)["\']', fh.read())
    return match.group(1) if match else "unknown"


# -- Project information -----------------------------------------------------

project = "ipie"
copyright = "2022-2026, The ipie Developers"
author = "The ipie Developers"
release = _read_version()
version = ".".join(release.split(".")[:2])

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "myst_parser",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"

templates_path = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

pygments_style = "sphinx"
todo_include_todos = True

# -- Autodoc / Napoleon ------------------------------------------------------

# Optional heavy dependencies that are not installed on Read the Docs.
# mpi4py is not mocked: ipie falls back to a serial FakeComm without it.
autodoc_mock_imports = [
    "cupy",
    "cupyx",
    "cuquantum",
    "torch",
    "pyscf",
    "trexio",
    "pyblock",
    "fqe",
    "jax",
    "jaxlib",
]

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autosummary_generate = False

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = True
# Avoid duplicate attribute descriptions for dataclass fields.
napoleon_use_ivar = True

# MyST (Markdown) settings for the developer guide.
myst_enable_extensions = ["colon_fence", "deflist"]
myst_heading_anchors = 3

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "h5py": ("https://docs.h5py.org/en/stable/", None),
}
intersphinx_timeout = 30

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "collapse_navigation": False,
    "navigation_depth": 4,
    "logo_only": True,
}
html_logo = "../../logo.png"
html_static_path = []
html_show_sourcelink = True

# -- Options for LaTeX / man / texinfo output --------------------------------

latex_documents = [
    (master_doc, "ipie.tex", "ipie Documentation", author, "manual"),
]
man_pages = [(master_doc, "ipie", "ipie Documentation", [author], 1)]
texinfo_documents = [
    (
        master_doc,
        "ipie",
        "ipie Documentation",
        author,
        "ipie",
        "Python-based auxiliary-field quantum Monte Carlo.",
        "Miscellaneous",
    ),
]
