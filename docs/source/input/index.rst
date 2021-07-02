Input Options
=============

PAUXY can be used either as a library or run with the usual "binary" + input file mode.
Here we will describe the input options necessary in both cases.

Input File
^^^^^^^^^^

A standard input file for pauxy is made up of a json dict like that below:

.. code-block:: json

    {
        "model": { },
        "qmc_options": { },
        "trial_wavefunction": { },
        "propagator": { },
        "estimates": {
            "mixed": { },
            "back_propagated": { },
            "itcf": { }
        }
    }

Model Options
^^^^^^^^^^^^^

Currently it is only possible to simulate the Hubbard model (1d or 2d), or a generic
system specified by a file containing the necessary one- and two-electron integrals.

Common Options
--------------

``name``
    type: string

    Name of model. Options: `Hubbard` or `Generic`.


Hubbard Model
-------------

``t``
    type: float

    Default 1.0.

    Hubbard hopping integral.

``U``
    type: float

    Default 1.0

    Hubbard U.

``nx``
    type: int

    Required.

    Number of lattice sites in x direction.

``ny``
    type: int

    Required.

    Number of lattice sites in y direction.

``nup``
    type: int

    Required.

    Number of spin up electrons.

``ndown``
    type: int

    Required.

    Number of spin down electrons.


Generic
-------

``integrals``
    type: string

    Required.

    Path to file containing one- and two-electron integrals. We assume an ascii FCIDUMP
    format as outlined, for example, `here <https://github.com/hande-qmc/fcidump/>`_. Note
    that we currently only can treat real integrals.

``decomposition``
    type: string

    Optional.

    Method by which we decompose two-electron integrals. Options:

        - ``cholesky`` Use cholesky decomposition. Default.
        - ``eigenvalue`` Use eigenvalue decomposition. Not implemented.

    threshold : float
        Cutoff for cholesky decomposition or minimum eigenvalue.
    verbose : bool
        Print extra information.

``nup``
    type: int

    Required.

    Number of spin up electrons.

``ndown``
    type: int

    Required.

    Number of spin down electrons.

QMC options
^^^^^^^^^^^
``dt``
    type: float

    Required.

    Timestep.

``nsteps``
    type: int

    Required.

    Total number of Monte Carlo steps to perform.

``nmeasure``
    type: int

    Required.

    Number of steps between measurement.

``nwalkers``
    type: int

    Required.

    Total number of walkers. If run in parallel then the number of walkers per core will
    be nwalkers / ncores.

``npop_control``
    type: int

    Default 10.

    Number of steps between population control.

``rng_seed``
    type: int

    Optional.

    Random number seed. Defaults to that calculated from system parameters via numpy.

Trial Wavefunction Options
^^^^^^^^^^^^^^^^^^^^^^^^^^

The trial wavefunction controls the constraint in CPMC and phaseless AFQMC and can
considerably affect results.

Common Options
--------------

The following options are common to all types of trial wavefunction.

``name``
    type: string

    Required.

    Currently we support the following options:

        - ``free_electron`` trial wavefunction found by diagonalising the corresponding
          one-electron part of the Hamiltonian.
        - ``UHF`` Constructs the trial wavefunction from the UHF solution to the Hubbard
          model. Only implemented for the Hubbard model.
        - ``multi_determinant`` Reads a multi-determinant trial wavefunction from an
          input file.
        - ``hartree_fock`` Constructs Hartree--Fock trial wavefunction assuming our
          one-electron basis is a set of molecular Hartree--Fock orbitals. Note this is
          the recommended (over the free_electron) option when simulating a generic model
          system.

``initial_wavefunction``
    type: string

    Optional.

    Specifies whether or not to use the trial wavefunction as the initial walker's Slater
    Determinant or to use a free-electron like state. Options `free_electron` or `trial`.
    Default `free_electron`.


Free Electron Options
---------------------

``read_in``
    type: string

    Optional.

    Input file to read trial wavefunction from. Default None. Assumes a single SD in
    either .npy file format or in column major fortran format which assumes complex
    numbers.

UHF options
-----------
``ninitial``
    type: int

    Optional.

    Number of random initial starting points for minimisation process. A simple attempt to
    attempt to find a global minimum. Default: 10.

``nconv``
    type: int

    Optional.

    Maximum number of steps in a single self consistent cycle. Default 5000.

``ueff``
    type: float

    Optional.

    Value of U to use for mean field Hamiltonian. Default 0.4.

``deps``
    type: float

    Optional.

    Convergence threshold for energy between self consistentcy cycles.

``alpha``
    type: float

    Optional.

    Mixing parameter. Default: 0.5.

``verbose``
    type: bool

    Optional.

    Print extra information on convergence rate. Default: false.

Multi-Determinant options
-------------------------

``type``
    type: string

    Required.

    Controls shape of SDs. Options: `GHF` or `UHF`. `UHF` SDs are of shape `(M,N)` while
    `GHF` SDs are of shape `(2M,N)` where `M` is the number of basis functions.

``orbitals``
    type: string

    Required.

    File containing orbitals. Currently assumes fortran (column major) format in data file
    with one (fortran fomatted) complex number per line.

``coefficients``
    type: string

    Required.

    File containing multi-determinant expansion coefficients. Expects one (fortran
    formatted) complex number per line.


Propagator Options
^^^^^^^^^^^^^^^^^^

``free_projection``
    type: bool

    Default False.

    Whether to perform free projection or not.

``hubbard_stratonovich``
    type: string

    Default None.

    Type of Hubbard-Stratonovich transformation to use. Options: `discrete`, `continuous`
    or `generic`. See ref:`theory/hubbard_stratonovich` for an explanation.

Estimator Options
^^^^^^^^^^^^^^^^^

By default we estimate basic (mixed) estimators. More sophisticated estimators can be
calculated using the optional ``mixed``, ``back_propagated`` and ``itcf`` dictionaries
through which we can estimate mixed estimates, back propagated quantities and imaginary
time correlation functions.

Common Options
--------------


``filename``
    type: string

    Optional.

    Output filename to which all estimators will be written (in hdf5 format.). Default:
    `estimates.0.h5`.

``overwrite``
    type: string

    Optional.

    If true and if no filename is specified then any further calculations will overwrite
    the default output file. If false then output will be written to a file whose index is
    one greater than the most recent output file. Default: true.

Mixed
-----
``rdm``
    type: bool

    Optional

    If true then the one-particle green's function is output to file. Default: false.

Back Propagated Options
-----------------------
``nback_prop``
    type: int

    Required.

    Number of back propagation steps to perform. This times the timestep determines the
    back propagation time.

``rdm``
    type: bool

    Optional

    If true then the one-particle green's function is output to file. Default: false.

ITCF Options
------------

``tmax``
    type: float

    Required.

    Maximum value of imaginary time to calculate ITCF to.

``stable``
    type: bool

    Optional.

    If true use the stabalised algorithm of Feldbacher and Assad. Default: true.

``mode``
    type: string / list

    Optional.

    How much of the ITCF to save to file. Options include:

        - ``full`` print full ITCF.
        - ``diagonal`` print diagonal elements of ITCF.
        - ``elements`` print select elements defined from list.

    Default: false.
``kspace``
    type: bool

    Optional.

    If true also evaluate correlation functions in momentum space. Default false.

.. toctree::
    :maxdepth: 2
    :glob:
