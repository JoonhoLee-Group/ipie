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

    python /path/to/ipie/tools/pyscf/pyscf_to_ipie.py -i 'scf.chk' -j 'input.json' --frozen-core 5

There are 7 electrons and 9 basis functions. Freezing the 5 core orbitals (1s2s2p) results in a CAS(5,8).

.. code-block:: json

	{
		"system": {
			"nup": 4,
			"ndown": 1
		},
		"hamiltonian": {
			"name": "Generic",
			"integrals": "hamiltonian.h5"
		},
		"qmc": {
			"dt": 0.005,
			"nwalkers": 640,
			"nsteps": 25,
			"blocks": 5000,
			"batched": true
		},
		"trial": {
			"filename": "wavefunction.h5"
		},
		"estimators": {}
	}

The above input options was updated with "nwalkers": 640 and "blocks": 5000.

Some reference numbers are:
	- UHF trial 	 				: 	-340.70469488 +/- 0.00005011
	- ROHF trial     				: 	-340.70476238 +/- 0.00005282
	- ROHF by QMCPack				: 	-340.704751   +/- 0.000053
	- UCCSD(T)		 				: 	-340.70490722

all-electron calculations:
	- UHF     		 				: 	-340.70786248 +/- 0.00004646
	- UHF using orthogonal AO		: 	-340.70786795 +/- 0.00005749
	- UCCSD(T)	 					:	-340.70799271
