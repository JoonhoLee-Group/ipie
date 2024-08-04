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

import numpy as np

np.random.seed(125)
from mpi4py import MPI

from ipie.qmc.afqmc import AFQMC
from ipie.systems import Generic
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial
# from ipie.addons.eph.trial_wavefunction.variational.toyozawa_variational import (
#     variational_trial_toyozawa,
# )
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.estimators.energy import EnergyEstimator

# System Parameters
nup = 2
ndown = 2
nelec = (nup, ndown)

# Hamiltonian Parameters
g = 2.0
t = 1.0
w0 = 4.0
nsites = 4
pbc = True

# Walker Parameters & Setup
comm = MPI.COMM_WORLD
nwalkers = 1000 // comm.size

# Setup initial guess for variational optimization
initial_electron = np.random.random((nsites, nup + ndown))
initial_phonons = np.ones(nsites) * 0.1

# System and Hamiltonian setup
system = Generic(nelec)
ham = HolsteinModel(g=g, t=t, w0=w0, nsites=nsites, pbc=pbc)
ham.build()

# Variational procedure - If Jax provided
# _, beta_shift, el_trial = variational_trial_toyozawa(
#     initial_phonons, initial_electron, ham, system
# )
# wavefunction = np.column_stack([beta_shift, el_trial])
wavefunction = np.load('wavefunction.npy')

# Setup trial
trial = ToyozawaTrial(
    wavefunction=wavefunction, w0=ham.w0, num_elec=[nup, ndown], num_basis=nsites
)
trial.set_etrial(ham)

# Setup walkers
walkers = EPhWalkers(
    initial_walker=wavefunction, nup=nup, ndown=ndown, nbasis=nsites, nwalkers=nwalkers
)
walkers.build(trial)

num_steps_per_block = 10
num_blocks = 10000
add_est = {
    "energy": EnergyEstimator(system=system, ham=ham, trial=trial),
}

seed = 125

# Note nwalkers specifies the number of walkers on each CPU
ephqmc = AFQMC.build(
    num_elec=nelec,
    hamiltonian=ham,
    trial_wavefunction=trial,
    walkers=walkers,
    num_walkers=nwalkers,
    seed=seed,
    num_steps_per_block=num_steps_per_block,
    num_blocks=num_blocks,
)
ephqmc.run(additional_estimators=add_est, verbose=False)
