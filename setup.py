import os
import sys

import numpy
from setuptools import find_packages, setup
from setuptools.extension import Extension

try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements

from Cython.Build import cythonize

# Giant hack to enable legacy code for CI
_build_legacy_extension = os.environ.get("BUILD_LEGACY_IPIE", False)
if _build_legacy_extension:
    extensions = [
        Extension(
            "ipie.legacy.estimators.ueg_kernels",
            ["ipie/legacy/estimators/ueg_kernels.pyx"],
            extra_compile_args=["-O3"],
            include_dirs=[numpy.get_include()],
        ),
    ]
    cythonized_extension = cythonize(
        extensions,
        include_path=[numpy.get_include()],
        compiler_directives={"language_level": sys.version_info[0]},
    )
else:
    extensions = []
    cythonized_extension = []


def load_requirements(fname):
    reqs = parse_requirements(fname, session="test")
    try:
        return [str(ir.req) for ir in reqs]
    except AttributeError:
        return [str(ir.requirement) for ir in reqs]


__version__ = ""
with open("ipie/__init__.py") as f:
    for line in f:
        if "__version__" in line:
            __version__ = line.split("=")[1].strip().strip('"')

setup(
    name="ipie",
    version=__version__,
    author="ipie developers",
    url="http://github.com/linusjoonho/ipie",
    packages=find_packages(exclude=["examples", "docs", "tests", "tools", "setup.py"]),
    license="Apache 2.0",
    description="Python implementations of Imaginary-time Evolution algorithms",
    python_requires=">=3.7.0",
    scripts=[
        "bin/ipie",
        "tools/extract_dice.py",
        "tools/reblock.py",
        "tools/fcidump_to_afqmc.py",
        "tools/pyscf/pyscf_to_ipie.py",
    ],
    ext_modules=cythonized_extension,
    install_requires=load_requirements("requirements.txt"),
    extras_require={
        "mpi": "mpi4py >= 3.0.0",
        "dev": ["mpi4py >= 3.0.0", "pyblock", "pyscf", "pytest-cov", "pytest-xdist"],
    },
    long_description=open("README.rst").read(),
)
