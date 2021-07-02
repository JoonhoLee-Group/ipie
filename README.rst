=====
PyQCQMC
=====

PyQCQMC is a collection of **P**\ ython implementations of **Q**\ uantum-**C**\ lassical
hybrid
or **Q**\ uantum **C**\ hemistry
**Q**\ uantum **M**\ onte **C**\ arlo algorithms with a focus on simplicity rather than speed.

PyQCQMC inherits a lot of QMC features from PAUXY.

.. image:: https://travis-ci.com/pyqcqmc-qmc/pyqcqmc.svg?branch=master
    :target: https://travis-ci.com/pyqcqmc-qmc/pyqcqmc

.. image:: http://readthedocs.org/projects/pyqcqmc/badge/?version=latest
    :target: http://pyqcqmc.readthedocs.io/en/latest/?badge=latest

.. image:: http://img.shields.io/badge/License-LGPL%20v2.1-blue.svg
    :target: http://github.com/fdmalone/pyqcqmc/blob/master/LICENSE

Features
--------
pyqcqmc can currently:

- estimate ground state properties of real (ab-initio) and model (Hubbard + UEG) systems.
- perform phaseless and constrained path AFQMC.
- calculate expectation values and correlation functions using back propagation.
- calculate imaginary time correlation functions.
- perform simple data analysis.

Installation
------------

Clone the repository

::

    $ git clone https://github.com/pyqcqmc-qmc/pyqcqmc.git

and run the following in the top-level pyqcqmc directory

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

pyqcqmc contains unit tests and some longer driver tests that can be run using pytest by
running:

::

    $ pytest -v

in the base of the repo. Some longer parallel tests are also run through the CI. See
travis.yml for more details.

.. image:: https://travis-ci.com/pyqcqmc-qmc/pyqcqmc.svg?branch=master
    :target: https://travis-ci.com/pyqcqmc-qmc/pyqcqmc

Documentation
-------------

Documentation and tutorials are available at
`readthedocs <https://pyqcqmc.readthedocs.org>`_.

.. image:: http://readthedocs.org/projects/pyqcqmc/badge/?version=latest
    :target: http://pyqcqmc.readthedocs.io/en/latest/?badge=latest
