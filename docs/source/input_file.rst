Input file reference
====================

This page documents the JSON input file consumed by the ``ipie`` command-line
launcher (``bin/ipie``)::

    mpirun -np N ipie input.json > output.dat

(or ``python /path/to/ipie/bin/ipie input.json`` from a source checkout).

Everything below is derived from the current driver code path,
:func:`ipie.qmc.calc.setup_calculation` → :func:`ipie.qmc.calc.read_input` →
:func:`ipie.qmc.calc.get_driver`. For the equivalent Python-level workflow
see :doc:`python_api`; for a worked example see :doc:`quickstart`.

How the file is read
--------------------

* MPI rank 0 parses the JSON with :func:`json.load` and broadcasts the
  resulting ``dict`` to all other ranks. A missing file raises
  :class:`FileNotFoundError` on every rank.
* The top level is an object whose keys name *sections* (``"system"``,
  ``"qmc"``, ...). Each section is itself an object of options.
* Options are looked up with :func:`ipie.utils.io.get_input_value`:
  the canonical key is tried first, then each alias in order, then the
  default. **Keys are case-sensitive** (they are plain ``dict`` lookups);
  ``"nwalkers"`` and ``"NWalkers"`` are different keys.
* A key whose value is JSON ``null`` is treated as absent (except
  ``verbosity`` and ``qmc.nmembers``, which are read with a plain
  ``dict.get`` and must be omitted rather than set to ``null``; a ``null``
  there raises :class:`TypeError`).
* Unknown keys are silently ignored. Nothing validates the file against a
  schema, so a misspelled option quietly falls back to its default.
* Passing ``--gpu`` on the command line sets the ``use_gpu`` configuration
  flag (see :doc:`advanced`); it is not controlled from the input file.
* The same dictionary can be handed directly to
  :func:`ipie.qmc.calc.get_driver` from Python, which is how
  :func:`ipie.qmc.calc.build_afqmc_driver` works.

Annotated example
-----------------

The file below extends the input that ``tools/pyscf/pyscf_to_ipie.py``
writes (via :func:`ipie.utils.io.write_json_input_file`; shown verbatim in
:doc:`quickstart`) with every option that the driver honours, spelled out and
commented. JSON has no comment syntax; the ``//`` lines are for this page
only.

.. code-block:: javascript

    {
        // Optional integer; 1 prints set-up information on rank 0, 0 is quiet,
        // >1 additionally echoes the system, hamiltonian, trial and
        // walkers sections.
        "verbosity": 1,
        "system": {
            "nup": 5,                       // number of alpha electrons (required)
            "ndown": 5                      // number of beta electrons (required)
        },
        "hamiltonian": {
            "name": "Generic",              // accepted but ignored
            "integrals": "hamiltonian.h5",  // HDF5 integral file (required)
            "symmetry": true                // pack Cholesky vectors (default true)
        },
        "qmc": {
            "nwalkers": 640,                // walkers PER MPI RANK (required)
            "dt": 0.005,                    // timestep in Ha^-1
            "nsteps": 25,                   // steps per block
            "blocks": 100,                  // number of blocks
            "stabilise_freq": 5,            // re-orthogonalise every 5 steps
            "pop_control_freq": 5,          // population control every 5 steps
            "pop_control_method": "pair_branch",
            "rng_seed": 7,                  // omit for a random seed
            "nmembers": 1,                  // ranks per shared-memory group
            "batched": true                 // must be true (or omitted)
        },
        "trial": {
            "filename": "wavefunction.h5",  // HDF5 wavefunction file (required)
            "ndets": 1,                     // multi-determinant trials only
            "ndets_props": 1,
            "ndet_chunks": 1
        },
        "walkers": {},                      // read, but not used (see below)
        "estimators": {}                    // written as {"filename": "estimates.0.h5"}
                                            // by the converter, but NOT read by the
                                            // driver (see below)
    }

Sections
--------

The driver looks up the following top-level keys. Every section defaults to
an empty object when absent.

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Key
     - Aliases
     - Purpose
   * - ``verbosity``
     - --
     - Integer output level (default ``1``).
   * - ``system``
     - ``model``
     - Electron counts.
   * - ``hamiltonian``
     - --
     - Integral file and packing.
   * - ``qmc``
     - ``qmc_options``
     - Propagation and population-control parameters
       (:class:`ipie.qmc.options.QMCOpts`).
   * - ``trial``
     - ``trial_wavefunction``
     - Wavefunction file and multi-determinant truncation.
   * - ``walkers``
     - ``walker``, ``walker_opts``
     - Parsed but has no effect in the current driver.
   * - ``estimators``
     - --
     - Not read by the current driver.

``verbosity``
-------------

A top-level integer, not a section. Default ``1``. On ranks other than 0 it
is forced to ``0`` after the sections have been located.

.. list-table::
   :header-rows: 1
   :widths: 12 88

   * - Value
     - Effect
   * - ``0``
     - No set-up output. The per-block estimator table is still printed on
       rank 0 (it is not controlled by ``verbosity``).
   * - ``1``
     - Print set-up information (integral reading timings, trial
       wavefunction summary, MPI group layout, defaults chosen, ...) and the
       final timing summary.
   * - ``>1``
     - As ``1``, plus echo the contents of the ``system``, ``hamiltonian``,
       ``trial`` and ``walkers`` sections as they are parsed.

``system``
----------

Read by :func:`ipie.systems.utils.get_system`. Only the generic (*ab
initio*) system type is supported by this driver.

.. list-table::
   :header-rows: 1
   :widths: 14 10 12 14 50

   * - Option
     - Type
     - Default
     - Aliases
     - Description
   * - ``nup``
     - int
     - *required*
     - --
     - Number of spin-up (alpha) electrons. If missing, rank 0 prints
       ``# Error: Number of electrons not specified.`` and calls
       ``sys.exit()`` (exit status 0); under MPI the remaining ranks do not
       exit there but fail separately with :class:`TypeError` when the
       system object is built from ``(None, None)``.
   * - ``ndown``
     - int
     - *required*
     - --
     - Number of spin-down (beta) electrons.
   * - ``name``
     - str
     - ``"Generic"``
     - --
     - Optional. Any value other than ``"Generic"`` raises
       :class:`ValueError`.

.. note::

   For backwards compatibility every key of ``system`` is also copied into
   the ``hamiltonian`` section before the latter is parsed (a ``name`` key in
   ``system`` never overrides one already present in ``hamiltonian``). Older
   input files therefore work with ``"integrals"`` placed under ``system``.
   New files should put it under ``hamiltonian``.

``hamiltonian``
---------------

Read in :func:`ipie.qmc.calc.get_driver` and passed to
:func:`ipie.hamiltonians.utils.get_hamiltonian`.

.. list-table::
   :header-rows: 1
   :widths: 14 10 12 20 44

   * - Option
     - Type
     - Default
     - Aliases
     - Description
   * - ``integrals``
     - str
     - *required*
     - --
     - Path to the HDF5 integral file (see :ref:`hamiltonian-h5`). A
       missing value raises ``ValueError("Hamiltonian filename not
       specified.")``.
   * - ``symmetry``
     - bool
     - ``true``
     - ``pack_chol``, ``pack_cholesky``
     - When ``true`` and shared memory is available, the packed
       upper-triangle Cholesky vectors (``L[i<=k, n]``) are built once per
       ``nmembers`` group and placed in an array shared by that group (with
       the default ``nmembers = 1`` every rank builds its own). ``false``
       disables the shared
       packed array; each rank then builds its own packed copy inside
       :class:`~ipie.hamiltonians.generic.GenericRealChol`. The full
       ``nbasis*nbasis`` vectors are kept in either case, so ``false`` never
       saves memory. Leave at the default.
   * - ``name``
     - str
     - --
     - --
     - Accepted for compatibility with the generated inputs
       (``"Generic"``) but never read.

The Hamiltonian is always a :class:`ipie.hamiltonians.generic.GenericRealChol`
object (built through the :func:`ipie.hamiltonians.generic.Generic` factory);
complex Cholesky, k-point and THC/ISDF Hamiltonians are only reachable from
Python (see :doc:`advanced`).

``qmc``
-------

Parsed by :class:`ipie.qmc.options.QMCOpts` and then copied into a
:class:`ipie.qmc.options.QMCParams` dataclass. Alias ``qmc_options`` is
accepted for the section name.

.. list-table::
   :header-rows: 1
   :widths: 18 8 10 22 42

   * - Option
     - Type
     - Default
     - Aliases
     - Description
   * - ``num_walkers``
     - int
     - *required*
     - ``nwalkers``
     - Number of walkers **per MPI rank**. The total population is
       ``num_walkers * (number of ranks)``. There is no default; omitting it
       makes walker construction fail.
   * - ``timestep``
     - float
     - ``0.005``
     - ``dt``
     - Imaginary-time step :math:`\Delta\tau` in :math:`\mathrm{Ha}^{-1}`.
   * - ``num_steps``
     - int
     - ``25``
     - ``nsteps``, ``steps``
     - Propagation steps per block. Estimators are accumulated over a block
       and written once per block.
   * - ``blocks``
     - int
     - ``1000``
     - ``num_blocks``, ``nblocks``
     - Number of blocks. Total steps = ``blocks * num_steps``.
   * - ``stabilise_freq``
     - int
     - ``5``
     - ``nstabilise``, ``reortho``
     - Re-orthogonalise (QR) walker orbitals every this many steps.
   * - ``pop_control_freq``
     - int
     - ``5``
     - ``npop_control``, ``pop_control``
     - Apply population control every this many steps.
   * - ``pop_control_method``
     - str
     - ``"pair_branch"``
     - ``pop_control``, ``population_control``
     - Population control algorithm; see :ref:`pop-control-methods`.
   * - ``rng_seed``
     - int
     - ``null``
     - ``random_seed``, ``seed``
     - Random seed. When absent rank 0 draws a random seed and broadcasts
       it. Each rank then seeds with ``rng_seed + rank``. Only the *input*
       value of ``rng_seed`` (``null`` when it was omitted) is recorded in
       the ``metadata`` of the estimator file; the randomly drawn seed is not
       serialised and is only printed by rank 0 to standard output as
       ``# random seed is N``. Keep the output file if you need to reproduce
       a run that used a random seed.
   * - ``nmembers``
     - int
     - ``1``
     - --
     - Number of MPI ranks per shared-memory group
       (:class:`ipie.utils.mpi.MPIHandler`). Must divide the total number of
       ranks. The Hamiltonian is read once per group and shared between its
       members.
   * - ``batched``
     - bool
     - ``true``
     - --
     - Must be ``true`` or omitted. ``false`` selects the removed
       non-batched code path and raises :class:`ValueError`.
   * - ``beta``
     - float
     - ``null``
     - --
     - Must be absent. Any value requests finite-temperature AFQMC, which
       is not available through this driver and raises :class:`ValueError`
       (use ``ipie.addons.thermal`` from Python instead).
   * - ``equilibration_time``
     - float
     - ``2.0``
     - ``tau_eqlb``
     - Parsed into the ``eqlb_time`` attribute of
       :class:`ipie.qmc.options.QMCOpts` but **not propagated** to :class:`ipie.qmc.options.QMCParams`; the JSON
       driver runs zero equilibration blocks and this option has no effect.

.. warning::

   ``pop_control`` is an alias of *both* ``pop_control_freq`` (an integer)
   and ``pop_control_method`` (a string). If you supply ``"pop_control":
   "comb"`` without also giving ``pop_control_freq``, the string is used as
   the frequency and propagation fails. Always use the canonical names
   ``pop_control_freq`` and ``pop_control_method``.

.. _pop-control-methods:

Population control methods
~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented in :class:`ipie.walkers.pop_controller.PopController`. Before
branching, all weights are rescaled so that the total weight equals the
target (total number of walkers).

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Value
     - Algorithm
   * - ``"pair_branch"`` (default)
     - Sort all walkers globally by weight and pair the lightest with the
       heaviest, the second lightest with the second heaviest, and so on. A
       pair is processed if the light walker's weight is below
       ``min_weight`` (fixed at ``0.1``) **or** the heavy walker's weight is
       above ``max_weight`` (fixed at ``4``); the individual weights are
       tested, not their sum. With probability
       :math:`w_{\mathrm{heavy}}/(w_{\mathrm{light}}+w_{\mathrm{heavy}})`
       the heavy walker is cloned (otherwise the light one), and both copies
       receive the weight :math:`(w_{\mathrm{light}}+w_{\mathrm{heavy}})/2`.
       Pairing stops at the first pair that satisfies neither condition.
   * - ``"comb"``
     - Comb (systematic) resampling of the whole population.
   * - ``"stochastic_reconfiguration"``
     - Stochastic reconfiguration.

Any other string prints ``Unknown population control method.`` on rank 0
and **performs no population control at all**; it does not abort.

``trial``
---------

Alias ``trial_wavefunction`` is accepted for the section name. Options are
passed to :func:`ipie.trial_wavefunction.utils.get_trial_wavefunction`.

.. list-table::
   :header-rows: 1
   :widths: 16 8 10 20 46

   * - Option
     - Type
     - Default
     - Aliases
     - Description
   * - ``filename``
     - str
     - ``""``
     - ``wfn_file``
     - Path to the HDF5 wavefunction file (see :ref:`wavefunction-h5`).
       Effectively required: the empty default cannot be opened.
   * - ``ndets``
     - int
     - ``1``
     - ``num_dets``
     - Particle-hole (CI-type) trials only: number of determinants kept from
       the file, in file order. ``-1`` keeps all of them. **The default is
       1**, so a multi-determinant file is used as a single-determinant
       trial unless this is set.
   * - ``ndets_props``
     - int
     - ``1``
     - ``num_dets_props``
     - Particle-hole trials only: number of determinants used to build the
       trial one-body density matrix, which sets the mean-field shift. The
       force bias and the local energy always use all ``ndets``. ``-1`` means
       ``ndets``; larger values are capped at ``ndets``.
   * - ``ndet_chunks``
     - int
     - ``1``
     - ``num_det_chunks``
     - Particle-hole trials only. ``1`` selects
       :class:`ipie.trial_wavefunction.particle_hole.ParticleHoleNonChunked`;
       ``>1`` selects the chunked
       :class:`ipie.trial_wavefunction.particle_hole.ParticleHole`, which
       evaluates the determinant expansion in ``ndet_chunks`` pieces to save
       memory.

The trial *type* is not an input option; it is inferred from the datasets in
the wavefunction file:

.. list-table::
   :header-rows: 1
   :widths: 28 30 42

   * - Dataset present (checked in this order)
     - Trial class
     - Notes
   * - ``occ_alpha``
     - :class:`~ipie.trial_wavefunction.particle_hole.ParticleHoleNonChunked` /
       :class:`~ipie.trial_wavefunction.particle_hole.ParticleHole`
     - CI expansion in a common orbital basis (from CASSCF / SHCI / ...).
       ``ndets``, ``ndets_props``, ``ndet_chunks`` apply.
   * - ``ci_coeffs`` (without ``occ_alpha``)
     - :class:`~ipie.trial_wavefunction.noci.NOCI`
     - Non-orthogonal CI: a linear combination of general determinants.
       ``ndets*`` options are ignored.
   * - ``psi_T_alpha`` (without ``ci_coeffs``)
     - :class:`~ipie.trial_wavefunction.single_det.SingleDet`
     - RHF/UHF/ROHF single determinant.
   * - ``Wavefunction`` group
     - one of the above
     - Legacy QMCPACK ``NOMSD``/``PHMSD`` format; electron counts are taken
       from the file, not from ``system``.

For particle-hole trials the number of electrons in the file may be smaller
than ``nup``/``ndown`` (e.g. a frozen-core CI expansion); the missing
"melting" core orbitals are inserted automatically and a message is printed.
For single-determinant trials the number of basis functions in the file must
equal that of the Hamiltonian (asserted).

.. note::

   NOCI trials are effectively **not loadable through the JSON driver** at
   present. :func:`ipie.trial_wavefunction.utils.get_trial_wavefunction`
   asserts ``nbasis == psi_T_alpha.shape[0]``, but
   :func:`ipie.utils.io.write_noci_wavefunction` stores the orbitals as
   ``(ndet, M, Na)``, so the check compares the number of determinants with
   the basis size and fails unless they happen to be equal. Build the
   :class:`~ipie.trial_wavefunction.noci.NOCI` object from Python instead
   (see :doc:`python_api`).

``walkers``
-----------

Aliases ``walker`` and ``walker_opts`` are accepted. The driver reads this
section, fills in ``pop_control`` (default ``"pair_branch"``) and its copy
``population_control``, and then **never uses the resulting dictionary**.
Walkers are constructed directly from ``qmc.num_walkers`` and the trial. In
particular a ``pop_control`` key placed here does *not* change the population
control method; use ``qmc.pop_control_method``.

``estimators``
--------------

**This section is not read by** :func:`ipie.qmc.calc.get_driver` **or by**
``bin/ipie``. The launcher calls ``afqmc.run(verbose=True)`` without an
``estimator_filename``, so :class:`ipie.estimators.handler.EstimatorHandler`
always uses its defaults:

* output file ``estimates.0.h5`` in the current working directory,
  **overwritten** if it exists;
* the single predefined observable ``"energy"``.

Keys such as ``"filename"``, ``"basename"``, ``"overwrite"`` or
``"observables"`` that appear in generated or historical input files are
therefore ignored. To choose a different output file or add observables,
drive the calculation from Python and pass ``estimator_filename`` and
``additional_estimators`` to :meth:`ipie.qmc.afqmc.AFQMC.run` (see
:doc:`python_api` and :doc:`examples`). Analysis of the output file is
described in :doc:`analysis`.

HDF5 file formats
-----------------

Both data files are plain HDF5 files written with :mod:`h5py`. The writers
live in :mod:`ipie.utils.io`; ``tools/pyscf/pyscf_to_ipie.py`` and
:func:`ipie.utils.from_pyscf.gen_ipie_input_from_pyscf_chk` produce them
from a PySCF checkpoint.

.. _hamiltonian-h5:

``hamiltonian.h5``
~~~~~~~~~~~~~~~~~~

Written by :func:`ipie.utils.io.write_hamiltonian`, read by
:func:`ipie.utils.io.read_hamiltonian` (via
:func:`ipie.hamiltonians.generic.read_integrals`). All arrays are real
``float64``; ``M`` is the number of (orthonormal) basis functions and ``X``
the number of Cholesky vectors.

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Dataset
     - Shape
     - Meaning
   * - ``hcore``
     - ``(M, M)``
     - One-body Hamiltonian :math:`h_{pq}` in the orthonormal basis (spin
       independent; the same matrix is used for both spins).
   * - ``LXmn``
     - ``(X, M, M)``
     - Cholesky vectors :math:`L^X_{mn}` with
       :math:`(mn|pq) \approx \sum_X L^X_{mn} L^X_{pq}`. The reader also
       accepts the dataset name ``chol`` and the transposed layout
       ``(M, M, X)``.
   * - ``e0``
     - scalar
     - Constant energy (nuclear repulsion plus any frozen-core
       contribution). The name ``ecore`` is accepted as an alternative.

The reader first tries the legacy QMCPACK layouts (``Hamiltonian/Factorized``
sparse and ``Hamiltonian/DenseFactorized/L`` dense, see
:func:`ipie.utils.io.from_qmcpack_sparse` and
:func:`ipie.utils.io.from_qmcpack_dense`) before falling back to the native
layout above, so either kind of file can be given as ``integrals``. Both
QMCPACK readers additionally require the ``Hamiltonian/Energies`` (constant
energy), ``Hamiltonian/dims`` (``nmo``, electron counts, number of Cholesky
vectors) and ``Hamiltonian/hcore`` datasets (the sparse reader also accepts
the older ``Hamiltonian/H1``/``H1_indx`` pair in place of ``hcore``).

.. _wavefunction-h5:

``wavefunction.h5``
~~~~~~~~~~~~~~~~~~~

Written by :func:`ipie.utils.io.write_wavefunction`, which dispatches on the
Python object it is given. ``M`` is the number of basis functions,
``Na``/``Nb`` the number of alpha/beta electrons and ``D`` the number of
determinants.

Single determinant (RHF/UHF/ROHF)

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Dataset
     - Shape
     - Meaning
   * - ``psi_T_alpha``
     - ``(M, Na)``
     - Occupied alpha orbitals of the trial, as columns. Required.
   * - ``psi_T_beta``
     - ``(M, Nb)``
     - Occupied beta orbitals. Optional; if absent ``psi_T_alpha`` is used
       for both spins (RHF). If present, ``phi0_beta`` must be present too
       (see below).
   * - ``phi0_alpha``
     - ``(M, Na)``
     - Alpha orbitals of the initial walker. **Required to be present** by
       the reader (a missing dataset raises :class:`KeyError`), but the
       driver discards it and starts walkers from ``psi_T``.
   * - ``phi0_beta``
     - ``(M, Nb)``
     - Beta initial walker orbitals. **Effectively required whenever**
       ``psi_T_beta`` **is given**: the reader loads ``psi_T_beta`` and
       ``phi0_beta`` inside a single ``try`` block, so a missing
       ``phi0_beta`` raises :class:`KeyError`, which is caught and silently
       makes the reader discard ``psi_T_beta`` and use ``psi_T_alpha`` for
       both spins. Like ``phi0_alpha`` its contents are then discarded by
       the driver.

Particle-hole multi-determinant (CI expansion)

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Dataset
     - Shape
     - Meaning
   * - ``ci_coeffs``
     - ``(D,)``
     - CI coefficients, ordered as the determinants should be truncated by
       ``ndets`` (typically by decreasing magnitude).
   * - ``occ_alpha``
     - ``(D, Na')``
     - Zero-based indices of the occupied alpha orbitals in each
       determinant. ``Na'`` may be smaller than ``Na`` for frozen-core
       expansions.
   * - ``occ_beta``
     - ``(D, Nb')``
     - Same for beta.

The orbitals themselves are the basis in which ``hamiltonian.h5`` is
expressed (e.g. the CASSCF natural orbitals), so no orbital coefficients are
stored.

Non-orthogonal CI (NOCI)

.. list-table::
   :header-rows: 1
   :widths: 16 22 62

   * - Dataset
     - Shape
     - Meaning
   * - ``ci_coeffs``
     - ``(D,)``
     - Expansion coefficients.
   * - ``psi_T_alpha``
     - ``(D, M, Na)``
     - Alpha orbitals of each determinant.
   * - ``psi_T_beta``
     - ``(D, M, Nb)``
     - Beta orbitals of each determinant.
   * - ``phi0_alpha``, ``phi0_beta``
     - ``(M, Na)``, ``(M, Nb)``
     - Written by the writer (first determinant by default); not read by
       the NOCI reader.

.. note::

   Because of the axis mix-up in the basis-size assertion described under
   ``trial`` above, a file in this layout can only be loaded by the JSON
   driver when ``D == M``. Use the Python API for NOCI trials.

Legacy QMCPACK wavefunctions (``Wavefunction/NOMSD`` or
``Wavefunction/PHMSD`` groups written by
:func:`ipie.utils.io.write_qmcpack_wfn`) are also accepted.

Generating the files
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    python /path/to/ipie/tools/pyscf/pyscf_to_ipie.py -i scf.chk -j input.json \
        [--mcscf] [--frozen-core N] [--ortho-ao] [-t 1e-5]

writes ``hamiltonian.h5``, ``wavefunction.h5`` and ``input.json``.
``--mcscf`` reads ``mcscf/ci_coeffs``, ``mcscf/occs_alpha`` and
``mcscf/occs_beta`` from the checkpoint and writes a particle-hole
wavefunction; ``--frozen-core N`` folds ``N`` core orbitals into ``hcore`` and
``e0`` and reduces ``nup``/``ndown`` in ``input.json`` accordingly. The
generated ``input.json`` uses ``nwalkers: 10`` and ``blocks: 100`` and
should be edited before a production run.

Legacy driver
-------------

``ipie --legacy input.json`` runs the deprecated driver in
``ipie.legacy.qmc.calc``. It shares the ``verbosity`` and
``qmc``/``qmc_options`` section lookup. In ``qmc`` it reads ``beta`` and
``batched``: a non-``null`` ``beta`` selects the legacy finite-temperature
driver (``ThermalAFQMC``), otherwise the legacy *non-batched* ``AFQMC`` driver
is used; ``batched`` is read but ignored. Otherwise it
accepts a different, undocumented option set (e.g. an ``estimates`` section,
``walkers.stack_size``, ``propagator`` options, model Hamiltonians selected
by ``system.name``). It is retained only for reproducing old results and is
not covered by this page.
