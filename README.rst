.. raw:: html

    <img src="https://github.com/linusjoonho/ipie/blob/main/logo.png" width="200px">
ipie stands for **I**\ntelligent **P**\ython-based **I**\maginary-time **E**\volution with a focus on simplicity and speed.

ipie inherits a lot of QMC features from pauxy.

.. image:: https://github.com/linusjoonho/ipie/workflows/CI/badge.svg
    :target: https://github.com/linusjoonho/ipie/workflows/CI/badge.svg

.. image:: http://readthedocs.org/projects/ipie/badge/?version=latest
    :target: http://ipie.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/License-Apache%20v2-blue.svg
    :target: http://github.com/linusjoonho/ipie/blob/master/LICENSE

Features
--------
ipie currently supports:

- estimation of the ground state energy of ab-initio systems using phaseless AFQMC with support for CPUs and GPUs.
- simple data analysis.
- other legacy functionalities available in pauxy such as the ground state and finite-temperature energies and properties (via backpropagation) of the ab initio, UEG, Hubbard, and Hubbard-Holstein models.

Installation
------------

Clone the repository

::

    $ git clone https://github.com/linusjoonho/ipie.git

and run the following in the top-level ipie directory

::

    $ pip install -r requirements.txt
    $ python setup.py build_ext --inplace
    $ python setup.py install

You may also need to set your PYTHONPATH appropriately.

Requirements
------------

* python (>= 3.6)
* numpy
* scipy
* h5py
* mpi4py
* cython
* pandas

Minimum versions are listed in the requirements.txt.
To run the tests you will need pytest.
To perform error analysis you will also need `pyblock <https://github.com/jsspencer/pyblock>`_.


Running the Test Suite
----------------------

ipie contains unit tests and some longer driver tests that can be run using pytest by
running:

::

    $ pytest -v

in the base of the repo. Some longer parallel tests are also run through the CI. See
`.github/workflows/ci2.yml` for more details.

.. image:: https://github.com/linusjoonho/ipie/workflows/CI/badge.svg
    :target: https://github.com/linusjoonho/ipie/workflows/CI/badge.svg

Documentation
-------------

Documentation and tutorials are available at
`readthedocs <https://ipie.readthedocs.org>`_.

.. image:: http://readthedocs.org/projects/ipie/badge/?version=latest
    :target: http://ipie.readthedocs.io/en/latest/?badge=latest
