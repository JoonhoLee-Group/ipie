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
`pyscf_to_pauxy.py` script found in `pauxy/tools/pyscf`.

.. code-block:: bash

    python /path/to/pauxy/tools/pyscf/pyscf_to_pauxy.py -i 'scf.chk' -j 'input.json'

You should find a file called `afqmc.h5` and pauxy input file `input.json` created from
information in `afqmc.h5`.

.. code-block:: json

    {
        "system": {
            "name": "Generic",
            "nup": 5,
            "ndown": 5,
            "integrals": "afqmc.h5"
        },
        "qmc": {
            "dt": 0.005,
            "nsteps": 10,
            "blocks": 1000,
            "nwalkers": 100,
            "pop_control_freq": 5
        },
        "trial": {
            "filename": "afqmc.h5"
        }
    }

The input options should be carefully updated, with particular attention paid to the
timestep `dt` and the total number of walkers `nwalkers`.

Run the AFQMC calculation by:

.. code-block:: bash

    mpirun -np N python /path/to/pauxy/bin/pauxy.py input.json

See the documentation for more input options and the converter:

.. code-block:: bash

    python /path/to/pauxy/tools/pyscf/pyscf_to_pauxy.py --help

The data can be analysed using

.. code-block:: bash

    python /path/to/pauxy/tools/reblock.py -s 1.0 -f estimates.0.h5

which will print a data table whose value for the total energy should be roughly
-5.38331344 +/- 0.0014386. This can be compared to value of -5.3819  +/- 0.0006 from the
Simons hydrogen chain benchmark `value`_. The results are roughly within error bars of
eachother, however we would typically recommend the use of a walker population of 1000 or
greater. The `-s` flag tells reblock.py to discard the first 1 a.u. of the simulation for
equilibration.

.. _value: https://github.com/simonsfoundation/hydrogen-benchmark-PRX/blob/master/N_10_OBC/R_1.6/AFQMC_basis-STO
