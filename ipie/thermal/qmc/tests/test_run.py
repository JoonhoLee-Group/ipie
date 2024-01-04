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
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee <linusjoonho@gmail.com>
#

import os
import sys
import tempfile
import uuid
import json
import h5py
import numpy

from pyscf import gto, scf, lo
from ipie.config import MPI
from ipie.systems.generic import Generic
from ipie.hamiltonians.utils import get_hamiltonian

from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.thermal.qmc.options import ThermalQMCParams
from ipie.thermal.qmc.thermal_afqmc import ThermalAFQMC
from ipie.thermal.estimators.energy import local_energy
from ipie.analysis.extraction import extract_test_data_hdf5, extract_observable

comm = MPI.COMM_WORLD
serial_test = comm.size == 1

# Unique filename to avoid name collision when running through CI.
if comm.rank == 0:
    test_id = str(uuid.uuid1())

else:
    test_id = None

test_id = comm.bcast(test_id, root=0)


def test_thermal_afqmc():
    ref_path = "reference_data/generic/"

    # Build `mol` object.
    nocca = 5
    noccb = 5
    nelec = nocca + noccb
    r0 = 1.75
    mol = gto.M(
        atom=[("H", i * r0, 0, 0) for i in range(nelec)],
        basis='sto-6g',
        unit='Bohr',
        verbose=5)

    # Build `scf` object.
    mf = scf.UHF(mol).run()
    mf.chkfile = 'scf.chk'
    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mf.stability(return_status=True)
    s1e = mol.intor("int1e_ovlp_sph")
    ao_coeff = lo.orth.lowdin(s1e)

    with h5py.File(ref_path + "generic_integrals.h5", "r") as fa:
        Lxmn = fa["LXmn"][:]
        nchol = Lxmn.shape[0]
        nbasis = Lxmn.shape[1]
    
    # Thermal AFQMC params.
    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 2
    seed = 7
    nsteps_per_block = 1
    nblocks = 10
    stabilize_freq = 10
    pop_control_freq = 1
    lowrank = False
    verbose = True

    params = ThermalQMCParams(
                num_walkers=nwalkers,
                total_num_walkers=nwalkers * comm.size,
                num_blocks=nblocks,
                num_steps_per_block=nsteps_per_block,
                timestep=timestep,
                beta=beta,
                num_stblz=stabilize_freq,
                pop_control_freq=pop_control_freq,
                rng_seed=seed)

    options = {
        "hamiltonian": {
            "name": "Generic",
            "integrals": ref_path + "generic_integrals.h5",
            "_alt_convention": False,
            "symmetry": False,
            "sparse": False,
            "mu": mu
        },
    }

    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')

    system = Generic(mol.nelec, verbose=verbose)
    hamiltonian = get_hamiltonian(system, options["hamiltonian"])
    trial = MeanField(hamiltonian, mol.nelec, beta, timestep, verbose=verbose)

    nbasis = trial.dmat.shape[-1]
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank, 
                                verbose=verbose)
    propagator = PhaselessGeneric(timestep, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)

    eloc = local_energy(hamiltonian, walkers)
    print(f'# Initial energy = {eloc[0]}')

    afqmc = ThermalAFQMC(system, hamiltonian, trial, walkers, propagator, params, verbose)
    afqmc.run(walkers, verbose)
    afqmc.finalise()
    afqmc.estimators.compute_estimators(afqmc.hamiltonian, afqmc.trial, afqmc.walkers)
    #numer_batch = afqmc.estimators["energy"]["ENumer"]
    #denom_batch = afqmc.estimators["energy"]["EDenom"]
    data_batch = extract_observable(afqmc.estimators.filename, "energy")
    print(data_batch)


if __name__ == '__main__':
    test_thermal_afqmc()

