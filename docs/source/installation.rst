Installation
============

ipie is a pure-Python package (with a few optional compiled helpers) that runs
on Linux and macOS. It requires Python 3.8 or newer (``python_requires>=3.8``
in ``setup.py``); the continuous-integration suite is exercised on Python
3.10, 3.11 and 3.12.

Installing from PyPI
--------------------

A pure-Python wheel (``py3-none-any``) is published on `PyPI
<https://pypi.org/project/ipie/>`_ and installs on Linux and macOS:

.. code-block:: bash

   pip install ipie

This installs the ``ipie`` package together with the required dependencies
listed below and puts a handful of command-line scripts on your ``PATH``
(``ipie``, ``pyscf_to_ipie.py``, ``reblock.py``, ``fcidump_to_afqmc.py`` and
``extract_dice.py``; they are listed under ``scripts`` in ``setup.py``, i.e.
they are copied files rather than ``console_scripts`` entry points).

Development installation
------------------------

To work on the code, or to use the very latest features, clone the repository
from GitHub and install it in editable mode:

.. code-block:: bash

   git clone https://github.com/JoonhoLee-Group/ipie.git
   cd ipie
   pip install -r requirements.txt
   pip install -e .

The build requires ``pip``, ``setuptools >= 42``, ``wheel``, ``numpy`` and
``Cython`` (see ``[build-system]`` in ``pyproject.toml``); ``pip`` fetches
these automatically.

.. note::

   The Cython extension for the legacy uniform-electron-gas kernels
   (``ipie.legacy.estimators.ueg_kernels``) is *not* built by default. Set
   ``BUILD_LEGACY_IPIE=1`` in the environment before running ``pip install -e .``
   if you need the ``ipie.legacy`` code paths, as the CI does.

Dependencies
------------

The required runtime dependencies (``requirements.txt``) are

===================== ============================================
Package               Purpose
===================== ============================================
``numpy >= 1.20.0``   array library
``scipy >= 1.3.0``    linear algebra and special functions
``h5py >= 3.0.0``     HDF5 input (integrals, trial) and output
``pandas``            tabular analysis of estimator data
``numba``             JIT-compiled CPU kernels
``plum-dispatch``     multiple dispatch used across the code base
``cython >= 0.29.0``  building the optional legacy extension
``pytest``            test suite
===================== ============================================

Optional extras
---------------

Four extras are defined in ``setup.py`` (``extras_require``). They can be
combined, e.g. ``pip install -e ".[mpi,torch]"``.

``mpi``
~~~~~~~

Parallel calculations are driven through `mpi4py
<https://mpi4py.readthedocs.io>`_ (``dev/mpi.txt``: ``mpi4py >= 3.0.0``):

.. code-block:: bash

   pip install -e ".[mpi]"       # or: pip install "ipie[mpi]"

``mpi4py`` needs a working MPI library on the machine. On a workstation the
simplest route is ``conda``:

.. code-block:: bash

   conda install openmpi          # or mpich

On a cluster, load the site MPI module first so that ``mpi4py`` is compiled
against it. See the `mpi4py installation guide
<https://mpi4py.readthedocs.io/en/stable/install.html>`_ for other options.

``torch``
~~~~~~~~~

The automatic-differentiation AFQMC add-on (``ipie.addons.adafqmc``, see
``examples/18-ad_afqmc``) is written on top of PyTorch (``dev/torch.txt``:
``torch >= 2.3.0``):

.. code-block:: bash

   pip install -e ".[torch]"

``gpu``
~~~~~~~

The cuTensorNet/cuQuantum backend used by the k-point and ISDF code paths
(``ipie.utils.cuquantum_backend``; ``dev/gpu.txt``:
``cuquantum-python-cu12 >= 26.1.0``). The wheel is specific to CUDA 12, needs
Python >= 3.11 and pulls in ``cupy-cuda12x``:

.. code-block:: bash

   pip install -e ".[gpu]"

For CUDA 13 install ``cuquantum-python-cu13`` by hand instead. The extra is
deliberately not part of ``dev`` because the CI runners have no CUDA toolkit.

``dev``
~~~~~~~

Developer tooling used by the CI (``dev/dev.txt``): ``pylint``, ``black``,
``flynt`` and ``pytest-xdist``. This extra also pulls in the ``mpi`` and
``torch`` requirements.

.. code-block:: bash

   pip install -e ".[dev]"

GPU support
-----------

GPU acceleration is provided through `CuPy <https://cupy.dev/>`_. Install a
CuPy wheel matching your CUDA toolkit by hand, for example

.. code-block:: bash

   pip install cupy-cuda12x       # pick the wheel for your CUDA version

or use the ``gpu`` extra described above, which brings in ``cupy-cuda12x``
together with the cuQuantum backend.

Multi-GPU runs communicate through MPI and require a *CUDA-aware* MPI build;
the ``openmpi`` packages on ``conda-forge`` are a convenient source.

GPU execution is switched on through ``ipie.config``. The switch must be set
**before any other ipie module is imported**, because
``ipie.utils.backend`` picks ``numpy`` or ``cupy`` as its array library at
import time. With the ``ipie`` command-line launcher pass the ``--gpu`` flag:

.. code-block:: bash

   mpirun -np 4 ipie --gpu input.json > output.dat

For your own Python scripts either export an environment variable before
starting Python,

.. code-block:: bash

   export IPIE_USE_GPU=1
   mpirun -np 4 python run_afqmc.py > output.dat

or set the option at the top of the script

.. code-block:: python

   from ipie.config import config
   config.update_option("use_gpu", True)

   # only now import the rest of ipie
   from ipie.qmc.calc import build_afqmc_driver

.. note::

   ``IPIE_USE_GPU`` is **not** honoured by the ``ipie`` launcher: ``bin/ipie``
   unconditionally calls ``config.update_option("use_gpu", options.use_gpu)``
   with the value of the ``--gpu`` flag, which overrides the environment
   variable. ``ipie.config`` also parses ``IPIE_USE_MIXED_PRECISION`` into
   the ``mixed_precision`` option, but nothing outside ``ipie.legacy`` reads
   that option, so setting it (or calling
   ``config.update_option("mixed_precision", True)``) currently has no
   effect on a calculation.

Optionally, the ISDF and k-point code paths can contract tensors with
cuTensorNet. Install ``cuquantum-python``; ``ipie.utils.cuquantum_backend``
imports the ``cuquantum.tensornet`` API and its error message recommends
``cuquantum>=26.1``, but no version check is performed. Without it ipie falls
back to ``cupy.einsum`` where possible.

Other optional packages
-----------------------

``pyscf``
   Needed to generate integrals and trial wavefunctions from a mean-field or
   MCSCF calculation, i.e. for :mod:`ipie.utils.from_pyscf` and the
   ``pyscf_to_ipie.py`` script (see :doc:`quickstart`).
``pyblock``
   Used by the ``--legacy`` analysis mode of ``reblock.py``
   (:func:`ipie.analysis.blocking.analyse_estimates`). The default,
   autocorrelation-based reblocking does not need it.
``trexio``
   Reading TREXIO files through :func:`ipie.utils.from_trexio.gen_ipie_from_trexio`
   (``examples/11-trexio``).
``fqe``
   Only used by the example script ``examples/14-fqe-wavefunction/run_afqmc.py``,
   which converts an OpenFermion-FQE wavefunction into an ipie multi-determinant
   trial; ipie itself never imports it.

The first three are imported lazily; ipie itself imports and runs without any
of them.

Building the Wicks helper library
---------------------------------

Multi-Slater-determinant trials can use optimised C routines for Wick's
theorem contractions. They live in ``ipie/lib/wicks`` and are compiled into a
shared library with CMake (``cmake_minimum_required(VERSION 3.4...3.18)``) and
a C compiler:

.. code-block:: bash

   cd ipie/lib/wicks
   cmake .
   make

or, as an out-of-source build (this is what the CI does),

.. code-block:: bash

   cd ipie/lib/wicks
   mkdir build && cd build
   cmake .. && make

Either way the result, ``libwicks_helper.so`` (``.dylib`` on macOS), is
written into ``ipie/lib/wicks`` where :mod:`ipie.lib.wicks.wicks_helper`
loads it with :mod:`ctypes`. Importing that module raises ``ImportError`` when
the library has not been built.

Running the test suite
----------------------

The tests are collected by ``pytest``; ``pytest.ini`` selects the
``unit``, ``driver`` and ``mpi`` markers by default. From the repository root:

.. code-block:: bash

   pytest                       # unit, driver and mpi-marked tests (single process)
   pytest -n auto               # in parallel, requires pytest-xdist

Additional categories are selected explicitly with ``-m``:

.. code-block:: bash

   IPIE_USE_GPU=1 pytest -m gpu # GPU unit tests (needs cupy and a GPU)
   pytest -m wicks              # tests of the compiled Wicks helper

``IPIE_USE_GPU=1`` is required for the GPU tests: it makes ``ipie.config``
select the ``cupy`` backend before the test modules import ipie (see
:doc:`dev/index`). Unset it again before running the CPU tests.

True multi-process tests are launched under MPI, for example

.. code-block:: bash

   mpirun -np 6 python -m pytest ipie/propagation/tests/test_generic_chunked.py
   mpirun -np 4 python -u ipie/qmc/tests/test_mpi_integration.py

``dev/run_tests.py`` wraps the linting (``--pylint``, ``--black``,
``--flynt``), MPI (``--mpi``), integration (``--integration``) and example
(``--examples``) jobs used in ``.github/workflows/ci.yml``.

Troubleshooting
---------------

Running without ``mpi4py``
   ipie does not strictly require MPI. If ``mpi4py`` cannot be imported,
   :mod:`ipie.config` substitutes a serial ``FakeComm`` communicator
   (:mod:`ipie.qmc.comm`) with ``rank == 0`` and ``size == 1``, so every
   script and example runs unchanged on a single process. You can check which
   communicator is active with

   .. code-block:: bash

      python -c "from ipie.config import MPI; print(MPI.COMM_WORLD)"

   which prints a ``FakeComm`` object in the serial case.

GPU option has no effect
   ``use_gpu`` must be set (via ``--gpu`` for the ``ipie`` launcher, or
   ``IPIE_USE_GPU=1`` / ``config.update_option`` for Python scripts) before
   ``ipie.utils.backend`` or any module that imports it is loaded. Setting it
   later in a script silently leaves the ``numpy`` backend in place. Note that
   the launcher ignores ``IPIE_USE_GPU``; it only looks at ``--gpu``.

Wicks helper cannot be imported
   Build the shared library as described above and make sure it ends up next
   to ``wicks_helper.py`` in ``ipie/lib/wicks``.
