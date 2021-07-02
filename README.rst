=====
PAUXY
=====

PAUXY is a collection of **P**\ ython implementations of **AUX**\ illiar\ **Y** field
quantum Monte Carlo algorithms with a focus on simplicity rather than speed.

.. image:: https://travis-ci.com/pauxy-qmc/pauxy.svg?branch=master
    :target: https://travis-ci.com/pauxy-qmc/pauxy

.. image:: http://readthedocs.org/projects/pauxy/badge/?version=latest
    :target: http://pauxy.readthedocs.io/en/latest/?badge=latest

.. image:: http://img.shields.io/badge/License-LGPL%20v2.1-blue.svg
    :target: http://github.com/fdmalone/pauxy/blob/master/LICENSE

Features
--------
PAUXY can currently:

- estimate ground state properties of real (ab-initio) and model (Hubbard + UEG) systems.
- perform phaseless and constrained path AFQMC.
- calculate expectation values and correlation functions using back propagation.
- calculate imaginary time correlation functions.
- perform simple data analysis.

Installation
------------

Clone the repository

::

    $ git clone https://github.com/pauxy-qmc/pauxy.git

and run the following in the top-level pauxy directory

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

Pauxy contains unit tests and some longer driver tests that can be run using pytest by
running:

::

    $ pytest -v

in the base of the repo. Some longer parallel tests are also run through the CI. See
travis.yml for more details.

.. image:: https://travis-ci.com/pauxy-qmc/pauxy.svg?branch=master
    :target: https://travis-ci.com/pauxy-qmc/pauxy

Documentation
-------------

Documentation and tutorials are available at
`readthedocs <https://pauxy.readthedocs.org>`_.

.. image:: http://readthedocs.org/projects/pauxy/badge/?version=latest
    :target: http://pauxy.readthedocs.io/en/latest/?badge=latest
