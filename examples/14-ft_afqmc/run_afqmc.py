import json
import numpy

from ueg import UEG
from ipie.config import MPI
from ipie.addons.thermal.qmc.calc import build_thermal_afqmc_driver
from ipie.analysis.extraction import extract_observable
from ipie.analysis.autocorr import reblock_by_autocorr

comm = MPI.COMM_WORLD

verbose = False if (comm.rank != 0) else True

# 1. Generate UEG integrals.
ueg_opts = {
            "nup": 1,
            "ndown": 1,
            "rs": 3,
            "ecut": 0.5,
            "thermal": True,
            "write_integrals": True
            }

ueg = UEG(ueg_opts, verbose=verbose)

if comm.rank == 0:
    ueg.build(verbose=verbose)

comm.barrier()

# 2. Build thermal AFQMC driver.
options = {
            'trial': {
                'name': 'one_body',
                },

            'walkers': {
                'lowrank': False,
                },
            
            'qmc': {
                'mu': 0.133579,
                'beta': 10,
                'timestep': 0.05,
                'nwalkers': 576 // comm.size,
                'nstack': 10,
                'seed': 7,
                'nblocks': 200,
                },
            }

afqmc = build_thermal_afqmc_driver(
            comm,
            nelec=ueg.nelec,
            hamiltonian_file='ueg_integrals.h5',
            seed=7,
            options=options,
            verbosity=verbose
        )

if verbose:
    print(f'\nThermal AFQMC options: \n{json.dumps(options, indent=4)}\n')
    print(afqmc.params)  # Inspect the qmc options.

# 3. Run thermal AFQMC calculation.
afqmc.run(verbose=verbose)
afqmc.finalise()
afqmc.estimators.compute_estimators(afqmc.hamiltonian, afqmc.trial, afqmc.walkers)

if comm.rank == 0:
    energy_data = extract_observable(afqmc.estimators.filename, "energy")
    number_data = extract_observable(afqmc.estimators.filename, "nav")

    print(f'filename: {afqmc.estimators.filename}')
    print(f'\nenergy_data: \n{energy_data}\n')
    print(f'number_data: \n{number_data}\n')
            
    y = energy_data["ETotal"]
    y = y[1:]  # Discard first 1 block.
    df = reblock_by_autocorr(y, verbose=verbose)
    print(df)
    print()
    
    y = number_data["Nav"]
    y = y[1:]  # Discard first 1 block.
    df = reblock_by_autocorr(y, verbose=verbose)
    print(df)
