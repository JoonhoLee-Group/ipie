.. image:: https://github.com/linusjoonho/ipie/blob/main/logo.png
    :width: 200

ipie stands for **I**\ntelligent **P**\ython-based **I**\maginary-time **E**\volution with a focus on simplicity and speed.

ipie inherits a lot of QMC features from pauxy.

.. image:: https://github.com/JoonhoLee-Group/ipie/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/JoonhoLee-Group/ipie/actions/workflows/ci.yml

.. image:: http://readthedocs.org/projects/ipie/badge/?version=latest
    :target: http://ipie.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/License-Apache%20v2-blue.svg
    :target: http://github.com/linusjoonho/ipie/blob/master/LICENSE

.. .. image:: https://codecov.io/gh/linusjoonho/ipie/branch/develop/graph/badge.svg
..     :target: https://codecov.io/gh/linusjoonho/ipie

.. image:: https://img.shields.io/badge/paper%20%28v0%29-arXiv%3A2209.04015-B31B1B
    :target: https://arxiv.org/abs/2209.04015

.. image:: https://img.shields.io/badge/paper%20%28v1%29-arXiv%3A2406.16238-B31B1B
    :target: https://arxiv.org/abs/2406.16238

Copyright by Joonho Lee (joonholee@g.harvard.edu)

ipie is a Python-based auxiliary-field quantum Monte Carlo (AFQMC) package, designed for simplicity and computational efficiency. The package has seen substantial improvements in modularity, functionality, and compatibility since its first release.

Key features include:
--------

- **Ground State Energy Estimation**: Calculate ground state energies of ab-initio systems with phaseless AFQMC.
- **Distributed Hamiltonian Simulations**: Run large-scale simulations distributed across multiple CPUs or GPUs, enabling calculations on systems too large for a single node or GPU card.
- **GPU Acceleration**: Support both CPU and GPU calculations, with GPU acceleration provided by CuPy/CUDA and CUDA-aware MPI.
- **Extended AFQMC Algorithms**: Includes free projection AFQMC, finite temperature AFQMC, AFQMC for electron-phonon systems, and automatic differentiation for property calculation.
- **Simple Data Analysis**
- **Other legacy features from pauxy**

For technical details, see our latest release papers:

- [J. Chem. Theory Comput., 2023, 19(1): 109-121](https://pubs.acs.org/doi/10.1021/acs.jctc.2c00934)
- [J. Chem. Phys. 161, 162502 (2024)](https://doi.org/10.1063/5.0225596)

Installation
------------

Linux and Mac OS wheels are available for installation via pip:

::

    $ pip install ipie

For development, clone the repository:

::

    $ git clone https://github.com/linusjoonho/ipie.git

Navigate to the top-level `ipie` directory and run:

::

    $ pip install -r requirements.txt
    $ pip install -e .

Requirements
------------

To build ipie with MPI support (via `mpi4py`), install with:

::

    $ pip install -e .[mpi]

This requires a working MPI installation on your machine, which can be installed via `conda`:

::

    conda install openmpi

Refer to the `mpi4py` `documentation <https://mpi4py.readthedocs.io/en/stable/install.html>`_ for alternative installation options.

For GPU support, `cupy` is required and can be installed as outlined on the `Cupy website <https://cupy.dev/>`_. For CUDA-aware MPI, consider `conda-forge`.

Running the Test Suite
----------------------

Unit tests and driver tests are included. To run all tests:

::

    $ pytest -v

More extensive parallel tests are executed in the CI; see `.github/workflows/ci.yml` for details.

.. image:: https://github.com/JoonhoLee-Group/ipie/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/JoonhoLee-Group/ipie/actions/workflows/ci.yml

Building Optimized C/C++ Code
-----------------------------

ipie also provides optimized implementations for certain functions involving multiple Slater determinant trials using C/C++ code, which can be built for enhanced performance. To compile these functions into a shared library, navigate to the `ipie/lib/wicks` folder and use `CMake` and `Make`:

::

    $ cmake .
    $ make

Documentation
-------------

Documentation and tutorials are available at `ReadTheDocs <https://ipie.readthedocs.org>`_.
