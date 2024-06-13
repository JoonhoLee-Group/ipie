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
"""Use a custom walker object.

In this example we will build a custom walker object. Note this requires a
little more interaction with ipie than previous examples as we can't rely on
factory methods so much.
"""

import numpy as np
from pyscf import gto, scf

from ipie.propagation.propagator import PhaselessGeneric
from ipie.qmc.options import QMCParams
from ipie.utils.mpi import MPIHandler

mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 10)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)
mf = scf.RHF(mol)
mf.chkfile = "scf.chk"
mf.kernel()

from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.backend import arraylib as xp


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


from ipie.estimators.energy import EnergyEstimator, local_energy_batch


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
        options={},
    ):
        super().__init__(
            system=system,
            ham=ham,
            trial=trial,
        )

    def compute_estimator(self, system, walkers, hamiltonian, trial):
        trial.calc_greens_function(walkers)
        # Need to be able to dispatch here
        energy = local_energy_batch(system, hamiltonian, walkers, trial)
        self._data["ENumer"] = xp.sum(walkers.weight * energy[:, 0].real)
        self._data["EDenom"] = xp.sum(walkers.weight)
        self._data["E1Body"] = xp.sum(walkers.weight * energy[:, 1].real)
        self._data["E2Body"] = xp.sum(walkers.weight * energy[:, 2].real)

        return self.data


from ipie.config import MPI

# Checkpoint integrals and wavefunction
# Running in serial but still need MPI World
from ipie.utils.from_pyscf import generate_hamiltonian

comm = MPI.COMM_WORLD

# Now let's build our custom AFQMC algorithm
num_walkers = 100
num_steps_per_block = 25
num_blocks = 10
timestep = 0.005

from ipie.qmc.afqmc import AFQMC

# 1. Build out system
from ipie.systems.generic import Generic

system = Generic(nelec=mol.nelec)

# 2. Build Hamiltonian

from ipie.utils.from_pyscf import generate_hamiltonian

ham = generate_hamiltonian(
    mol,
    mf.mo_coeff,
    mf.get_hcore(),
    mf.mo_coeff,  # should be optional
)

# 3. Build trial wavefunction
from ipie.utils.from_pyscf import generate_wavefunction_from_mo_coeff

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
trial.build()
trial.half_rotate(ham)

np.random.seed(7)
from ipie.walkers.uhf_walkers import UHFWalkers

mpi_handler = MPIHandler()


class CustomUHFWalkers(UHFWalkers):
    def __init__(self, initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler):
        super().__init__(initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler)

    def reortho(self):
        print("customized reortho called")
        return super().reortho()


walkers = CustomUHFWalkers(
    np.hstack([orbs, orbs]),
    system.nup,
    system.ndown,
    ham.nbasis,
    num_walkers,
    mpi_handler,  # initial_walkers
)
params = QMCParams(
    num_walkers=num_walkers,
    total_num_walkers=num_walkers * comm.size,
    num_blocks=num_blocks,
    num_steps_per_block=num_steps_per_block,
    timestep=timestep,
    rng_seed=7,
)

propagator = PhaselessGeneric(params.timestep)
propagator.build(ham, trial, walkers, mpi_handler)

afqmc = AFQMC(
    system,
    ham,
    trial,
    walkers,
    propagator,
    mpi_handler,
    params,
)
estimator = NoisyEnergyEstimator(system=system, ham=ham, trial=trial)
add_est = {"energy": estimator}
afqmc.run(additional_estimators=add_est)

from ipie.analysis.extraction import extract_observable

qmc_data = extract_observable(afqmc.estimators.filename, "energy")
y = qmc_data["ETotal"]
y = y[1:]  # discard first 1 block

from ipie.analysis.autocorr import reblock_by_autocorr

df = reblock_by_autocorr(y, verbose=1)
# print(df.to_csv(index=False))

# assert np.isclose(df.at[0,'ETotal_ac'], -5.3360473872294305)
# assert np.isclose(df.at[0,'ETotal_error_ac'], 0.011931730085308796)
