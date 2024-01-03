
.. image:: https://github.com/linusjoonho/ipie/blob/main/logo.png
    :width: 200

ipie stands for **I**\ntelligent **P**\ython-based **I**\maginary-time **E**\volution with a focus on simplicity and speed.

ipie inherits a lot of QMC features from pauxy.

.. image:: https://github.com/linusjoonho/ipie/workflows/CI/badge.svg
    :target: https://github.com/linusjoonho/ipie/workflows/CI/badge.svg

.. image:: http://readthedocs.org/projects/ipie/badge/?version=latest
    :target: http://ipie.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/License-Apache%20v2-blue.svg
    :target: http://github.com/linusjoonho/ipie/blob/master/LICENSE

.. .. image:: https://codecov.io/gh/linusjoonho/ipie/branch/develop/graph/badge.svg
..     :target: https://codecov.io/gh/linusjoonho/ipie

.. image:: https://img.shields.io/badge/paper%20%28v0%29-arXiv%3A2209.04015-B31B1B
    :target: https://arxiv.org/abs/2209.04015

Copyright by Joonho Lee (joonholee@g.harvard.edu)

Features
--------
ipie currently supports:

- estimation of the ground state energy of ab-initio systems using phaseless AFQMC with support for CPUs and GPUs.
- simple data analysis.
- other legacy functionalities available in pauxy such as the ground state and finite-temperature energies and properties (via backpropagation) of the ab initio, UEG, Hubbard, and Hubbard-Holstein models.

Installation
------------

Linux and Mac OS wheels are available for installation via pip

::

    $ pip install ipie

For develpment you can instead clone the repository

::

    $ git clone https://github.com/linusjoonho/ipie.git

and run the following in the top-level ipie directory

::

    $ pip install -r requirements.txt
    $ pip install -e .

Requirements
------------

To build ipie with MPI support (via mpi4py) do:

::

    $ pip install -e .[mpi]

Note that mpi4py requires a working MPI installation to be built on your
machine. This  it is often the trickiest dependency to setup correctly.

One of the easiest ways (if you are using pip to install ipie wheels) is via
conda:

::

    conda install openmpi

which will just install the OpenMPI library. 
We refer users to the mpi4py
`documentation <https://mpi4py.readthedocs.io/en/stable/install.html>`_ for
alternative ways of building mpi4py and the required MPI library.

Further requirements are listed in requirements.txt.

GPU Support
-----------
Cupy is is required when running calculations on GPUs which
can be install following the instructions `here <https://cupy.dev/>`_ .

Cuda aware MPI may be installed via conda-forge.

Running the Test Suite
----------------------

ipie contains unit tests and some longer driver tests that can be run using pytest by
running:

::

    $ pytest -v

in the base of the repo. Some longer parallel tests are also run through the CI. See
`.github/workflows/ci.yml` for more details.

.. image:: https://github.com/linusjoonho/ipie/workflows/CI/badge.svg
    :target: https://github.com/linusjoonho/ipie/workflows/CI/badge.svg

Documentation
-------------

Documentation and tutorials are available at
`readthedocs <https://ipie.readthedocs.org>`_.

.. image:: http://readthedocs.org/projects/ipie/badge/?version=latest
    :target: http://ipie.readthedocs.io/en/latest/?badge=latest
