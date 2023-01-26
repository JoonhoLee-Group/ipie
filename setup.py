import sys

from setuptools import find_packages, setup

try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements

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
    python_requires=">=3.6.0",
    scripts=[
        "bin/ipie",
        "tools/extract_dice.py",
        "tools/reblock.py",
        "tools/fcidump_to_afqmc.py",
        "tools/pyscf/pyscf_to_ipie.py",
    ],
    install_requires=load_requirements("requirements.txt"),
    long_description=open("README.rst").read(),
)
