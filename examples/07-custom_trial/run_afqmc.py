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
# Author: Fionn Malone <fmalone@google.com>
#

import numpy as np
from pyscf import gto, scf

from ipie.analysis.autocorr import reblock_by_autocorr

# We can extract the qmc data as as a pandas data frame like so
from ipie.analysis.extraction import extract_observable
from ipie.estimators.energy import EnergyEstimator, local_energy_batch
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import arraylib as xp
from ipie.utils.from_pyscf import generate_hamiltonian, generate_wavefunction_from_mo_coeff

mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 10)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)
mf = scf.RHF(mol)
mf.chkfile = "scf.chk"
mf.kernel()


# Let's make a trial wavefunction than injects noise into the overlap evaluateio
# We will inherit from SingleDet trial class to avoid some boilerplate.
# For odder trials you should inherit from TrialWavefunctionBase
class NoisySingleDet(SingleDet):
    def __init__(
        self,
        wavefunction,
        num_elec,
        num_basis,
        verbose=False,
        noise_level=1e-12,
    ):
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        self._noise_level = noise_level

    def calc_overlap(self, walkers) -> np.ndarray:
        ovlp = super().calc_overlap(walkers)
        noise = np.random.normal(scale=self._noise_level, size=ovlp.size)
        return ovlp * (1 + noise)

    def calc_greens_function(self, walkers, build_full=False) -> np.ndarray:
        greens = super().calc_greens_function(walkers, build_full)
        noise = np.random.normal(scale=self._noise_level, size=greens.size).reshape(greens.shape)
        return greens * (1 + noise)

    def calc_force_bias(self, hamiltonian, walkers, mpi_handler=None) -> np.ndarray:
        force_bias = super().calc_force_bias(hamiltonian, walkers, mpi_handler)
        noise = np.random.normal(scale=self._noise_level, size=force_bias.size).reshape(
            force_bias.shape
        )
        return force_bias * (1 + noise)


# Need to define a custom energy estimator as currently we don't have multiple dispatch setup.
# Just derive from the usual energy estimator and overwrite this in the driver.
class NoisyEnergyEstimator(EnergyEstimator):
    def __init__(
        self,
        comm=None,
        qmc=None,
        system=None,
        ham=None,
        trial=None,
        verbose=False,
    ):
        super().__init__(
            system=system,
            ham=ham,
            trial=trial,
        )

    def compute_estimator(self, system, walkers, hamiltonian, trial, istep=1):
        trial.calc_greens_function(walkers)
        # Need to be able to dispatch here
        energy = local_energy_batch(system, hamiltonian, walkers, trial)
        self._data["ENumer"] = xp.sum(walkers.weight * energy[:, 0].real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        self._data["E1Body"] = xp.sum(walkers.weight * energy[:, 1].real)
        self._data["E2Body"] = xp.sum(walkers.weight * energy[:, 2].real)

        return self.data


# Checkpoint integrals and wavefunction
# Running in serial but still need MPI World

# Now let's build our custom AFQMC algorithm
num_walkers = 100
num_steps_per_block = 25
num_blocks = 10
timestep = 0.005


# 1. Build Hamiltonian
ham = generate_hamiltonian(
    mol,
    mf.mo_coeff,
    mf.get_hcore(),
    mf.mo_coeff,  # should be optional
)

# 2. Build trial wavefunction
orbs = generate_wavefunction_from_mo_coeff(
    mf.mo_coeff,
    mf.mo_occ,
    mf.mo_coeff,  # Make optional argument
    mol.nelec,
)
num_basis = mf.mo_coeff[0].shape[-1]
trial = NoisySingleDet(
    np.hstack([orbs, orbs]),
    mol.nelec,
    num_basis,
    noise_level=1e-3,
)
# TODO: would be nice if we didn't need to do this.
trial.build()
trial.half_rotate(ham)

afqmc = AFQMC.build(
    mol.nelec,
    ham,
    trial,
    num_walkers=num_walkers,
    num_steps_per_block=num_steps_per_block,
    num_blocks=num_blocks,
    timestep=timestep,
    seed=59306159,
)
add_est = {"energy": NoisyEnergyEstimator(system=Generic(mol.nelec), ham=ham, trial=trial)}
afqmc.run(additional_estimators=add_est)

qmc_data = extract_observable(afqmc.estimators.filename, "energy")
y = qmc_data["ETotal"]
y = y[1:]  # discard first 1 block

df = reblock_by_autocorr(y, verbose=1)
