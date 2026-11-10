ipie documentation
==================

ipie stands for **I**\ ntelligent **P**\ ython-based **I**\ maginary-time
**E**\ volution. It is an auxiliary-field quantum Monte Carlo (AFQMC) package
for *ab initio* quantum chemistry and model Hamiltonians, written in Python with
a focus on simplicity and computational efficiency.

Key features
------------

- **Ground-state energies** of *ab initio* systems with phaseless AFQMC.
- **Distributed Hamiltonians**: large calculations can be spread across many
  CPUs or GPUs, enabling systems that do not fit on a single node or GPU card.
- **GPU acceleration** via CuPy/CUDA and CUDA-aware MPI.
- **Multi-determinant trials**: particle-hole (MCSCF/SHCI) trial wavefunctions
  with optimized Wick's-theorem kernels (see :doc:`advanced`).
- **Extended algorithms**: free-projection AFQMC, finite-temperature AFQMC,
  automatic-differentiation AFQMC for properties, GHF walkers, periodic
  (k-point) Cholesky and ISDF Hamiltonians, and isometric-THC (iTHC)
  factorised molecular Hamiltonians.
- **Simple data analysis** tools for reblocking and error estimation.

The code is hosted on `GitHub <https://github.com/JoonhoLee-Group/ipie>`_ and
released on `PyPI <https://pypi.org/project/ipie/>`_. If you use ipie, please
cite the release papers listed in :doc:`bibliography`.

.. toctree::
   :maxdepth: 2
   :caption: Getting started

   installation
   quickstart

.. toctree::
   :maxdepth: 2
   :caption: User guide

   input_file
   python_api
   analysis
   advanced
   examples

.. toctree::
   :maxdepth: 2
   :caption: Background

   theory
   bibliography

.. toctree::
   :maxdepth: 2
   :caption: Reference

   api/index
   dev/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
