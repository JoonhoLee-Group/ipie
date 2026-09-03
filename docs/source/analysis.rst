Analysing results
=================

An AFQMC calculation produces a time series of block averages. Turning that
series into an energy with a reliable error bar requires three steps: identify
and discard the equilibration period, account for the correlation between
successive blocks, and average. This page describes what ipie writes out,
how to load it into `pandas <https://pandas.pydata.org>`_, and the tools in
:mod:`ipie.analysis` and ``tools/reblock.py`` that automate the error
analysis.

What a run produces
-------------------

Standard output
~~~~~~~~~~~~~~~

Rank 0 prints a table with one row per block. The header line and the first
few rows look like this (values shortened for readability; in the real output
every number is printed in scientific notation with 16 digits after the
decimal point)::

              Block                   Weight             WeightFactor             HybridEnergy                   ENumer                   EDenom                   ETotal                   E1Body                   E2Body
                  0   6.4000000000000000e+02   6.4000000000000000e+02   0.0000000000000000e+00  -3.4123317203485680e+03   6.4000000000000000e+02  -5.3317683130446375e+00  -1.0224831392713812e+01   4.8930630796691753e+00
                  1   6.4056270262847788e+02   6.3899304380906920e+02  -5.3444960443893420e+00  -3.4258270011121011e+03   6.4000000000000000e+02  -5.3528546892376580e+00  -1.0217926384616452e+01   4.8650716953787941e+00
                  2   ...

Lines beginning with ``#`` contain set-up information (the git commit, the MPI
configuration, the parameters, the trial wavefunction energy) and, after
``afqmc.finalise(verbose=True)``, the timing breakdown. The columns are:

``Block``
   The block index. Block ``0`` is special: it is evaluated on the *initial*
   walker population before any propagation, so for a single-determinant
   trial its ``ETotal`` equals the trial energy. Real blocks start at ``1``
   and are separated by ``num_steps_per_block`` propagation steps, i.e. by
   ``num_steps_per_block * timestep`` in imaginary time.

``Weight``
   The total walker weight, summed over all MPI tasks and averaged over the
   steps in the block. Population control rescales the weights so that this
   fluctuates around the total number of walkers.

``WeightFactor``
   The same average but using the weights as they were *before* the last
   population-control rescaling. Comparing it with ``Weight`` shows how much
   the population grows or shrinks between population-control events, which
   in turn reflects how well the energy shift tracks the ground-state energy.

``HybridEnergy``
   The weighted average of the walkers' hybrid energies over the block. It is
   used as the energy shift :math:`E_{\text{shift}}` in the propagator for the
   next block. It is a useful diagnostic (it should settle near ``ETotal``)
   but it is not an unbiased estimate of the energy.

``ENumer`` and ``EDenom``
   The numerator :math:`\sum_w w \, E_L(\phi_w)` and denominator
   :math:`\sum_w w` of the mixed energy estimate, evaluated on the last step of
   the block and summed over all tasks. ``EDenom`` is a single-step total
   weight, which is why it differs from ``Weight``.

``ETotal``
   The mixed estimate of the total energy for this block,
   ``ENumer / EDenom``, in Hartree. This is the column you will analyse.

``E1Body`` and ``E2Body``
   The one-body (kinetic plus electron--nucleus plus any frozen-core
   contribution, including the constant ``ecore``) and two-body pieces of
   ``ETotal``, normalised in the same way. They sum to ``ETotal``.

Custom estimators that set ``print_to_stdout = True`` add their own columns
after ``E2Body``.

If an explicit equilibration phase was requested (``num_eq_blocks > 0`` in
:meth:`ipie.qmc.afqmc.AFQMC.build`) its blocks are printed first, and the
``Block`` counter restarts from ``1`` when the production phase begins. In the
HDF5 file the rows are simply stored in the order they were produced.

The HDF5 estimator file
~~~~~~~~~~~~~~~~~~~~~~~

Everything printed to stdout, plus any array-valued estimators, is written to
an HDF5 file, by default ``estimates.0.h5`` in the working directory (set
``estimator_filename`` in ``afqmc.run()`` to change it). Note that the
``"estimators": {"filename": ...}`` block that ``pyscf_to_ipie.py`` writes into
``input.json`` is ignored by the current driver: ``ipie input.json`` always
writes ``estimates.0.h5``. The layout is:

* ``metadata``: a JSON string with a serialised copy of the driver, so the
  input parameters, Hamiltonian and trial information, the seed and the
  environment (``sys_info``, present when the driver was built with
  ``verbose=True``) are stored alongside the data. Read it with
  :func:`ipie.analysis.extraction.get_metadata`.
* ``block_size_1/data/000000000``, ``000000001``, ...: two-dimensional
  complex datasets holding one row per block. Rows are written as they are
  produced into preallocated datasets of ``estimator_buffer_size`` rows (1000
  by default, an ``ipie.config.config`` option), one dataset per chunk. Each
  row contains the three walker
  properties (``Weight``, ``WeightFactor``, ``HybridEnergy``) followed by the
  flattened buffers of each estimator in the order they were registered.
* ``block_size_1/max_block/<chunk>``: the index of the last row written in
  each chunk, so partially written files (from a run that is still going or
  was killed) can be read.
* ``block_size_1/names/<estimator>``, ``shape/``, ``size/``, ``offset/``,
  ``scalar/``: bookkeeping describing where each estimator's data sits in a
  row and how to reshape it.

You rarely need this detail because the extraction functions below hide it,
but it means the file can be read by any HDF5 tool and can be inspected
while the calculation is still running.

Loading data into pandas
------------------------

The main entry point is :func:`ipie.analysis.extraction.extract_observable`::

   extract_observable(filename, name="energy", block_idx=1)

For the energy (and any other estimator with ``scalar_estimator = True``) it
returns a :class:`pandas.DataFrame` with the walker properties and the
estimator's named entries as columns, and one row per block starting with
block 0:

.. code-block:: python

   from ipie.analysis.extraction import extract_observable

   data = extract_observable("estimates.0.h5", "energy")
   print(data.columns.tolist())
   # ['Weight', 'WeightFactor', 'HybridEnergy', 'ENumer', 'EDenom', 'ETotal', 'E1Body', 'E2Body']
   print(data["ETotal"].head())

For array-valued estimators (``scalar_estimator = False``, see
:doc:`python_api`) it instead returns a NumPy array of shape
``(nblocks + 1,) + estimator.shape`` in which every block has already been
divided by its own denominator, i.e. it holds per-block mixed estimates:

.. code-block:: python

   rdm = extract_observable("estimates.0.h5", "1RDM")   # e.g. shape (101, 2, nbasis, nbasis)

Other useful functions in :mod:`ipie.analysis.extraction`:

* :func:`~ipie.analysis.extraction.extract_data_from_textfile` parses the
  table from a redirected stdout file (``output.dat``) into a DataFrame with
  a ``Block`` column, for cases where only the text output was kept. It stops
  at the ``End Time`` line printed by ``finalise``.
* :func:`~ipie.analysis.extraction.get_metadata` and
  :func:`~ipie.analysis.extraction.get_param` return the JSON metadata as a
  dictionary or a single entry from it (``get_param(f, ["params", "timestep"])``).
* :func:`~ipie.analysis.extraction.extract_hdf5_data` returns the raw
  ``(data, info)`` pair underlying ``extract_observable``.
* :func:`~ipie.analysis.extraction.extract_data`,
  :func:`~ipie.analysis.extraction.extract_data_sets`,
  :func:`~ipie.analysis.extraction.extract_mixed_estimates`,
  :func:`~ipie.analysis.extraction.extract_bp_estimates` and
  :func:`~ipie.analysis.extraction.extract_rdm` read the older
  ``basic/energies``, ``back_propagated/...`` layout produced by the legacy
  code in ``ipie.legacy``. They do not apply to ``estimates.0.h5`` files
  written by the current :class:`~ipie.estimators.handler.EstimatorHandler`.
  The thermal and free-projection add-ons write the same ``block_size_1``
  layout as the current handler and are read with ``extract_observable``
  (the free-projection add-on ships its own copy in
  :mod:`ipie.addons.free_projection.analysis.extraction`).

Once you have a DataFrame the usual pandas idioms apply: ``data.to_csv(...)``,
``data["ETotal"].plot()``, or concatenating several files with
``pandas.concat``.

Equilibration
-------------

The walkers start from the trial determinant (or from a perturbation of it),
and the first blocks of the run track the decay of excited-state
contributions rather than sampling the ground state. These blocks must be
discarded. There is no automatic criterion; plot ``ETotal`` against the block
index and choose the point after which it fluctuates around a constant:

.. code-block:: python

   import matplotlib.pyplot as plt

   data = extract_observable("estimates.0.h5", "energy")
   data["ETotal"].plot(marker=".", linestyle="none")
   plt.xlabel("Block"); plt.ylabel("ETotal / Ha")
   plt.show()

   start = 10                                 # chosen by inspection
   etotal = data["ETotal"].values[start:]

For most molecules a few tens of blocks of 25 steps at a timestep of 0.005
(i.e. one to several Hartree\ :sup:`-1` of imaginary time) suffice, but poor
trial wavefunctions and small gaps lengthen the transient. When in doubt,
verify that the final energy is stable with respect to the number of blocks
discarded. Two more observations from the same plot are worth making: the
``HybridEnergy`` and ``Weight`` columns should also have settled, and a slow
drift that never flattens out usually indicates a population that is too small
or a timestep that is too large rather than incomplete equilibration.

ipie also offers an *explicit* equilibration phase (``num_eq_blocks``,
``eq_timestep``, ``eq_num_steps_per_block`` in
:meth:`~ipie.qmc.afqmc.AFQMC.build`) during which a larger timestep and more
frequent stabilisation and population control can be used to reach the ground
state faster. Blocks from that phase are still written to the file and must
still be discarded in the analysis.

Estimating the statistical error
--------------------------------

Successive blocks are correlated, because the walkers only move a little in
``num_steps_per_block`` steps and because population control couples the
whole population. The naive standard error, ``etotal.std() / sqrt(len(etotal))``,
therefore underestimates the true uncertainty, often by a large factor. Two
standard remedies are implemented.

Autocorrelation analysis
~~~~~~~~~~~~~~~~~~~~~~~~

:func:`ipie.analysis.autocorr.reblock_by_autocorr` is the default and
needs only NumPy and pandas::

   reblock_by_autocorr(y, name="ETotal", verbose=False)

It estimates the integrated autocorrelation time :math:`\tau` of the series
``y`` with the FFT-based estimator of Goodman and Weare (with Sokal's
automatic window), groups the data into blocks of ``ceil(tau)`` consecutive
samples, which are then approximately independent, and returns the mean and
the standard error of those block averages:

.. code-block:: python

   from ipie.analysis.autocorr import reblock_by_autocorr

   result = reblock_by_autocorr(etotal, verbose=True)
   print(result)
   #     ETotal_ac  ETotal_error_ac  ETotal_nsamp_ac  ac
   # 0   -5.382214         0.000631               45   4

The columns are the mean (``ETotal_ac``), its one-sigma standard error
(``ETotal_error_ac``), the number of independent blocks that went into the
average (``ETotal_nsamp_ac``) and the block size in units of the original
blocks (``ac``). With ``verbose=True`` the function also prints :math:`\tau`
estimated from the full series and from successively halved subsets, which
tells you whether the estimate has converged; if :math:`\tau` keeps growing as
more data is included, the run is too short for a reliable error bar. As a
rule of thumb you want ``ETotal_nsamp_ac`` to be at least 20--30.

:func:`ipie.analysis.blocking.reblock_minimal` wraps this for one or more
files, discarding the first ``start_block`` rows of each::

   reblock_minimal(files, start_block=0, verbose=False)

.. code-block:: python

   from ipie.analysis.blocking import reblock_minimal

   summary = reblock_minimal(["estimates.0.h5", "estimates.1.h5"], start_block=10)
   print(summary.to_string(index=False))

Each file is analysed separately (there is one row per file, with a
``filename`` column). Files may be HDF5 estimator files or redirected text
output; anything whose name does not contain ``.h5`` is parsed with
``extract_data_from_textfile``.

Reblocking with pyblock
~~~~~~~~~~~~~~~~~~~~~~~

The classic alternative is Flyvbjerg--Petersen reblocking: repeatedly average
neighbouring pairs of blocks and look for the plateau in the standard error.
ipie uses the `pyblock <https://github.com/jsspencer/pyblock>`_ package for
this. It is an optional dependency (``pip install pyblock``) that is only
needed here and for the ``--legacy`` mode of ``reblock.py``. Applied to a
DataFrame it gives a table of the error at every reblocking level and a
recommended level:

.. code-block:: python

   import pyblock

   data_len, reblocked, covariance = pyblock.pd_utils.reblock(data[["ETotal"]].iloc[start:])
   print(reblocked)                                               # full table
   print(pyblock.pd_utils.reblock_summary(reblocked.loc[:, "ETotal"]))   # optimal block

The functions :func:`ipie.analysis.blocking.reblock_mixed` and
:func:`ipie.analysis.blocking.analyse_estimates` in :mod:`ipie.analysis.blocking`
perform this analysis on the legacy file layout and group results by the
simulation parameters stored in the metadata; they are kept for backwards
compatibility. When both methods are applicable they should agree within their
own uncertainties. The autocorrelation estimate is usually the more robust of
the two for the relatively short series that AFQMC produces, which is why it is
the default.

Averaging array-valued estimators
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a density matrix or another array observable returned by
``extract_observable`` the same logic applies element-wise. A simple and
usually adequate approach is to discard the equilibration rows and use the
block size ``ac`` determined from the energy:

.. code-block:: python

   import numpy as np

   rdm = extract_observable("estimates.0.h5", "1RDM")[start:]          # (nsamp, 2, nbasis, nbasis)
   ac = int(reblock_by_autocorr(etotal)["ac"][0])
   nblk = len(rdm) // ac
   blocked = rdm[: nblk * ac].reshape((nblk, ac) + rdm.shape[1:]).mean(axis=1)
   rdm_mean = blocked.mean(axis=0)
   rdm_err = blocked.std(axis=0, ddof=1) / np.sqrt(nblk)

Remember that these are mixed estimates (see the warning in
:doc:`python_api`).

The ``reblock.py`` command-line tool
------------------------------------

``tools/reblock.py`` in the repository exposes the analysis above from the
shell. When ipie is installed with pip the script is also on ``PATH``, so
``reblock.py -b 10 -f estimates.0.h5`` works from any directory (as used in
:doc:`quickstart`). The everyday invocation is

.. code-block:: bash

   python /path/to/ipie/tools/reblock.py -b 10 -f estimates.0.h5

which discards the first 10 blocks and prints the ``reblock_minimal`` table::

        ETotal_ac  ETotal_error_ac  ETotal_nsamp_ac  ac        filename
     -5.38221417       0.00063142               45   4  estimates.0.h5

Several files can be given after ``-f`` (a glob pattern in quotes is also
accepted), and a redirected stdout file works just as well as the HDF5 file:

.. code-block:: bash

   python /path/to/ipie/tools/reblock.py -b 10 -f output.dat
   python /path/to/ipie/tools/reblock.py -b 10 -f "run_*/estimates.0.h5"

The options are:

``-f FILE [FILE ...]``
   Files to analyse. Required.

``-b N, --block-start N``
   Discard the first ``N`` blocks (default ``0``). This is the equilibration
   control for the default analysis.

``-v, --verbose``
   Print the autocorrelation-time convergence table in addition to the
   summary.

``--legacy``
   Run the pyblock-based :func:`~ipie.analysis.blocking.analyse_estimates`
   on files in the legacy layout. With it, ``-s TIME, --start TIME`` selects
   the equilibration cut-off as an imaginary time in Hartree\ :sup:`-1`
   instead of a block count. ``-m, --multi-sim`` is accepted for backwards
   compatibility but has no effect: files are always grouped and averaged by
   the parameters found in their metadata. ``-t, --average-tau`` prints the
   energy as a function of imaginary time but currently exits with an error
   after printing the table.

``--free-proj``
   Legacy free-projection error analysis (ratio of averaged ``ENumer`` and
   ``EDenom`` over runs) for files in the old ``basic/energies`` layout only.
   It does not read output from the current free-projection add-on; use
   :func:`ipie.addons.free_projection.analysis.jackknife.jackknife_ratios`
   for that (see :doc:`advanced`).

Recommended workflow
--------------------

Putting the pieces together, the procedure used throughout the examples
(``examples/01-simple/README.rst``) is:

1. Run the calculation, keeping the stdout: ``mpirun -np N python run_afqmc.py > output.dat``
   or ``mpirun -np N ipie input.json > output.dat``.
2. Look at the ``ETotal`` column (in ``output.dat`` or by plotting the
   DataFrame from ``extract_observable``) and decide how many blocks to
   discard.
3. Run ``python tools/reblock.py -b <nblocks> -f estimates.0.h5`` (or
   ``reblock_by_autocorr`` from Python) and check that ``ETotal_nsamp_ac`` is
   comfortably large and that the autocorrelation time has converged
   (``-v``).
4. Report ``ETotal_ac`` with ``ETotal_error_ac`` as the one-sigma statistical
   error.

For the H\ :sub:`10` chain of the first example this gives an energy that
should agree with the published benchmark value of −5.3819(6) Ha within the
combined error bars. Remember that the statistical error is only part of the
story: the phaseless constraint and the finite timestep introduce systematic
errors that do not show up in the error bar. The former is controlled by the
quality of the trial wavefunction and the latter by repeating the calculation
at two or three timesteps and extrapolating; both are discussed in
:doc:`theory` and :doc:`advanced`. Finally, the population-control bias
decreases with the total number of walkers, which is why the examples
recommend a total population of at least several hundred and preferably a
thousand or more.
