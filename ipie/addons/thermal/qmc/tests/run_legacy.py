# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

import json
import os
import sys
import tempfile
import uuid
import numpy
from typing import Tuple, Union

from ipie.config import MPI
from ipie.addons.thermal.utils.testing import build_driver_ueg_test_instance
from ipie.addons.thermal.utils.legacy_testing import build_legacy_driver_ueg_test_instance
from ipie.analysis.extraction import (
        extract_test_data_hdf5, 
        extract_data,
        extract_observable, 
        extract_mixed_estimates)

comm = MPI.COMM_WORLD
serial_test = comm.size == 1

# Unique filename to avoid name collision when running through CI.
if comm.rank == 0:
    test_id = str(uuid.uuid1())

else:
    test_id = None

test_id = comm.bcast(test_id, root=0)


def run_legacy():
    # UEG params.
    nup = 1
    ndown = 1
    nelec = (nup, ndown)
    rs = 3
    ecut = 0.5

    # Thermal AFQMC params.
    mu = 2.158847
    beta = 1
    timestep = 0.05
    nwalkers = 12 // comm.size
    # Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
    nsteps_per_block = 1
    nblocks = 10
    pop_control_freq = 1
    pop_control_method = "pair_branch"
    #pop_control_method = "comb"
    lowrank = False
    lowrank_thresh = 1e-14
    stack_size = 10
    optimised = False
    
    verbose = False if (comm.rank != 0) else True
    debug = True
    seed = 7
    numpy.random.seed(seed)
    
    with tempfile.NamedTemporaryFile() as tmpf1:
        options = {
                    'nelec': nelec,
                    'mu': mu,
                    'beta': beta,
                    'timestep': timestep,
                    'nwalkers': nwalkers,
                    'seed': seed,
                    'nsteps_per_block': nsteps_per_block,
                    'nblocks': nblocks,
                    'pop_control_freq': pop_control_freq,
                    'pop_control_method': pop_control_method,
                    'lowrank': lowrank,
                    'lowrank_thresh': lowrank_thresh,
                    'stack_size': stack_size,
                    'optimised': optimised,

                    "ueg_opts": {
                        "nup": nup,
                        "ndown": ndown,
                        "rs": rs,
                        "ecut": ecut,
                        "thermal": True,
                        "write_integrals": False,
                        "low_rank": lowrank
                    },

                    "estimators": {
                        "filename": tmpf1.name, # For legacy.
                    },
                }

        # ---------------------------------------------------------------------
        # Test.
        # ---------------------------------------------------------------------
        afqmc = build_driver_ueg_test_instance(options, seed, debug, verbose)
        
        # ---------------------------------------------------------------------
        # Legacy.
        # ---------------------------------------------------------------------
        print('\n------------------------------')
        print('Running Legacy ThermalAFQMC...')
        print('------------------------------')
        legacy_afqmc = build_legacy_driver_ueg_test_instance(
                        afqmc.hamiltonian, comm, options, seed, verbose)
        legacy_afqmc.run(comm=comm)
        legacy_afqmc.finalise(verbose=False)
        legacy_afqmc.estimators.estimators["mixed"].update(
            legacy_afqmc.qmc,
            legacy_afqmc.system,
            legacy_afqmc.hamiltonian,
            legacy_afqmc.trial,
            legacy_afqmc.walk,
            0,
            legacy_afqmc.propagators.free_projection)
        legacy_mixed_data = extract_mixed_estimates(legacy_afqmc.estimators.filename)

        enum = legacy_afqmc.estimators.estimators["mixed"].names
        legacy_energy_numer = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.enumer]
        legacy_energy_denom = legacy_afqmc.estimators.estimators["mixed"].estimates[enum.edenom]
        
        if verbose:
            print(f'\nThermal AFQMC options: \n{json.dumps(options, indent=4)}\n')



if __name__ == '__main__':
    run_legacy()

