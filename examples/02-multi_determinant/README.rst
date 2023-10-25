Simple PYSCF Workflow
=====================

In this example we will go through the basic steps necessary to generate AFQMC input from
a pyscf casscf calculation N2.

The pyscf scf script is given below (scf.py in the current directory):

.. code-block:: python

    import h5py

    from pyscf import ao2mo, fci, gto, lib, mcscf, scf


    nocca = 4
    noccb = 2
    mol = gto.M(
        atom=[("N", 0, 0, 0), ("N", (0, 0, 3.0))],
        basis="ccpvdz",
        verbose=3,
        spin=nocca - noccb,
        unit="Bohr",
    )
    mf = scf.RHF(mol)
    mf.chkfile = "scf.chk"
    ehf = mf.kernel()
    M = 6
    N = 6
    mc = mcscf.CASSCF(mf, M, N)
    mc.chkfile = "scf.chk"
    e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
    # tol specifies the ci coefficient tolerance.
    # note it may be helpful to sort the ci coefficents if you want to truncate the
    # expansion.
    coeff, occa, occb = zip(
        *fci.addons.large_ci(fcivec, M, (nocca, noccb), tol=1e-8, return_strs=False)
    )
    # Need to write wavefunction to checkpoint file.
    with h5py.File("scf.chk", 'r+') as fh5:
        fh5['mcscf/ci_coeffs'] = coeff
        fh5['mcscf/occs_alpha'] = occa
        fh5['mcscf/occs_beta'] = occb


The important point is to specify the `chkfile` option.

Once the scf converges we need to generate the wavefunction and integrals using the
`pyscf_to_ipie.py` script found in `ipie/tools/pyscf`.

.. code-block:: bash

    python /path/to/ipie/tools/pyscf/pyscf_to_ipie.py -i 'scf.chk' -j 'input.json' --mcscf

Important here is the `--mcscf` flag which will tell the converter to read the mcscf
mo_coefficients and also look for the msd wavefunction.

You should find a file called `wavefunction.h5` which defines the multi-Slater expansion, `hamiltonian.h5` and ipie input file `input.json` shown as below.
We can run AFQMC with
.. code-block:: bash

    mpirun -np N python /path/to/ipie/bin/ipie input.json > output.dat

or, by running the `run_afqmc.py` of the current folder.

.. code-block:: json

    {
        "system": {
            "nup": 8,
            "ndown": 6
        },
        "hamiltonian": {
            "name": "Generic",
            "integrals": "hamiltonian.h5"
        },
        "qmc": {
            "dt": 0.005,
            "nwalkers": 640,
            "nsteps": 25,
            "blocks": 1000,
            "batched": true,
            "pop_control_freq": 5,
            "stabilise_freq": 5
        },
        "trial": {
            "filename": "wavefunction.h5",
            "compute_trial_energy": true
        },
        "estimators": {
            "filename": "estimates.0.h5"
        }
    }

Note we added the option `compute_trial_energy` to the input file. It is **always
recommended** to check the variational energy of the trial wavefunction you use to ensure
there is no translation errors. Currently the algorithm to compute this variational energy
is sub-optimal so this option is defaulted to false. One can control the number of
determinants used to compute the variational energy with the `ndets`
option. It can be helpful to set this to a value smaller than the number of determinants
in the trial wavefunction. One should also add the `ndets_props` which is the number od dets used for propagation.
