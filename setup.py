import numpy
from setuptools import find_packages, setup
from setuptools.extension import Extension
import sys
import versioneer
try:
    from pip._internal.req import parse_requirements
except ImportError:
    from pip.req import parse_requirements
from Cython.Build import cythonize

extensions = [
        Extension("pie.estimators.ueg_kernels",
                  ["pie/estimators/ueg_kernels.pyx"],
		  include_dirs=[numpy.get_include()])
        ]

def load_requirements(fname):
    reqs = parse_requirements(fname, session="test")
    try:
        return [str(ir.req) for ir in reqs]
    except AttributeError:
        return [str(ir.requirement) for ir in reqs]

setup(
    name='pie',
    version=versioneer.get_version(),
    author='PIE developers',
    url='http://github.com/linusjoonho/pie',
    packages=find_packages(exclude=['examples', 'docs', 'tests', 'tools', 'setup.py']),
    license='Apache 2.0',
    description='Python implementations of Imaginary-time Evolution algorithms',
    python_requires=">=3.6.0",
    install_requires=load_requirements("requirements.txt"),
    long_description=open('README.rst').read(),
    ext_modules = cythonize(extensions, include_path=[numpy.get_include()],
                            compiler_directives={'language_level':
                                                 sys.version_info[0]})
)
