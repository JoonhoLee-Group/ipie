=====
PIE
=====

PIE stands for **P**\ython-based **I**\maginary-time **E**\volution with a focus on simplicity and speed.

PIE inherits a lot of QMC features from pauxy.

.. image:: https://github.com/linusjoonho/pie/actions/workflows/ci2.yml/badge.svg
    :target: https://github.com/inusjoonho/pie/actions/workflows/ci2.yml

.. image:: http://readthedocs.org/projects/pie/badge/?version=latest
    :target: http://pie.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/License-Apache%20v2-blue.svg
    :target: http://github.com/linusjoonho/pie/blob/master/LICENSE

Features
--------
pie can currently:

- estimate ground state properties of real (ab-initio) and model (Hubbard + UEG) systems.
- perform phaseless and constrained path AFQMC.
- calculate expectation values and correlation functions using back propagation.
- calculate imaginary time correlation functions.
- perform simple data analysis.

Installation
------------

Clone the repository

::

    $ git clone https://github.com/linusjoonho/pie.git

and run the following in the top-level pie directory

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

pie contains unit tests and some longer driver tests that can be run using pytest by
running:

::

    $ pytest -v

in the base of the repo. Some longer parallel tests are also run through the CI. See
`.github/workflows/ci2.yml` for more details.

.. image:: https://github.com/linusjoonho/pie/actions/workflows/ci2.yml/badge.svg
    :target: https://github.com/inusjoonho/pie/actions/workflows/ci2.yml

Documentation
-------------

Documentation and tutorials are available at
`readthedocs <https://pie.readthedocs.org>`_.

.. image:: http://readthedocs.org/projects/pie/badge/?version=latest
    :target: http://pie.readthedocs.io/en/latest/?badge=latest
