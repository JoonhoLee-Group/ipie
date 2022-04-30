Simple PYSCF Workflow
=====================

In this example we will go through how to generate frozen-core AFQMC input from
a pyscf scf calculation for a P atom.

The pyscf scf script is given below (scf.py in the current directory):

.. code-block:: python

    from pyscf import gto, scf, cc

	atom = gto.M(atom='P 0 0 0',
				 basis='6-31G',
				 verbose=4,
				 spin=3,
				 unit='Bohr')
				 
	mf = scf.UHF(atom)
    mf.chkfile = 'scf.chk'
    mf.kernel()

The important point is to specify the `chkfile` option.

Once the scf converges we need to generate the wavefunction and integrals using the
`pyscf_to_ipie.py` script found in `ipie/tools/pyscf`.

.. code-block:: bash

    python /path/to/ipie/tools/pyscf/pyscf_to_ipie.py -i 'scf.chk' -j 'input.json' -c 5,8

There are 7 electrons and 9 basis functions. Freezing the 5 core orbitals (1s2s2p) results in a CAS(5,8).
You should find a file called `afqmc.h5` and ipie input file `input.json` created from
information in `afqmc.h5`.

.. code-block:: json

	{
		"system": {
			"nup": 4,
			"ndown": 1
		},
		"hamiltonian": {
			"name": "Generic",
			"integrals": "afqmc.h5"
		},
		"qmc": {
			"dt": 0.005,
			"nwalkers": 640,
			"nsteps": 25,
			"blocks": 5000,
			"batched": true
		},
		"trial": {
			"filename": "afqmc.h5"
		},
		"estimators": {}
	}

The above input options was updated with "nwalkers": 640 and "blocks": 5000.

Run the AFQMC calculation by:

.. code-block:: bash

    mpirun -np N python /path/to/ipie/bin/ipie input.json

See the documentation for more input options and the converter:

.. code-block:: bash

    python /path/to/ipie/tools/pyscf/pyscf_to_ipie.py --help

The data can be analysed using

.. code-block:: bash

    python /path/to/ipie/tools/reblock.py -s 1.0 -f estimates.0.h5

which will print a data table whose value for the total energy should be roughly:
	- UHF trial 	 				: 	-340.70469488 +/- 0.00005011
	- ROHF trial     				: 	-340.70476238 +/- 0.00005282
	- ROHF by QMCPack				: 	-340.704751   +/- 0.000053
	- UCCSD(T)		 				: 	-340.70490722

all-electron calculations:
	- UHF     		 				: 	-340.70786248 +/- 0.00004646 
	- UHF using orthogonal AO		: 	-340.70786795 +/- 0.00005749
	- UCCSD(T)	 					:	-340.70799271
