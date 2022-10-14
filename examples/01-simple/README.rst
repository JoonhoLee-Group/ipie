Simple PYSCF Workflow
=====================

In this example we will go through the basic steps necessary to generate AFQMC input from
a pyscf scf calculation for a simple H10 chain.

The pyscf scf script is given below (scf.py in the current directory):

.. code-block:: python

    from pyscf import gto, scf, cc

    atom = gto.M(atom=[('H', 1.6*i, 0, 0) for i in range(0,10)],
                 basis='sto-6g',
                 verbose=4,
                 unit='Bohr')
    mf = scf.UHF(atom)
    mf.chkfile = 'scf.chk'
    mf.kernel()

The important point is to specify the `chkfile` option.

Once the scf converges we need to generate the wavefunction and integrals using the
`pyscf_to_ipie.py` script found in `ipie/tools/pyscf`.

.. code-block:: bash

    python /path/to/ipie/tools/pyscf/pyscf_to_ipie.py -i 'scf.chk' -j 'input.json'

You should find a file called `afqmc.h5` and ipie input file `input.json` created from
information in `afqmc.h5`.

.. code-block:: json

    {
        "system": {
            "nup": 5,
            "ndown": 5
        },
        "hamiltonian": {
            "name": "Generic",
            "integrals": "hamiltonian.h5"
        },
        "qmc": {
            "dt": 0.005,
            "nwalkers": 640,
            "nsteps": 25,
            "blocks": 100,
            "batched": true,
            "pop_control_freq": 5,
            "stabilise_freq": 5
        },
        "trial": {
            "filename": "wavefunction.h5"
        },
        "estimators": {
            "filename": "estimates.0.h5"
        }
    }

The input options should be carefully updated, with particular attention paid to the
timestep `dt` and the total number of walkers `nwalkers`.

Run the AFQMC calculation by:

.. code-block:: bash

    mpirun -np N python /path/to/ipie/bin/ipie input.json > output.dat

See the documentation for more input options and the converter:

.. code-block:: bash

    python /path/to/ipie/tools/pyscf/pyscf_to_ipie.py --help

The data can be analysed using

.. code-block:: bash

    python /path/to/ipie/tools/reblock.py -b 10 -f output.dat 

which will print a data table whose value for the total energy which should be
comparable to -5.3819  +/- 0.0006 from the Simons hydrogen chain benchmark
`value`_. The results should roughly be within error bars of eachother, however we
would typically recommend the use of a walker population of 1000 or greater. The
`-b` flag tells reblock.py to discard the first 10 blocks of the simulation for
equilibration. This is not automatic and a visual inspection of the ETotal
column is typically necessary to determine the number of blocks to discard.

.. _value: https://github.com/simonsfoundation/hydrogen-benchmark-PRX/blob/master/N_10_OBC/R_1.6/AFQMC_basis-STO
