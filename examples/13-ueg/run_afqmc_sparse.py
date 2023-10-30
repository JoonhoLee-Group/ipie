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

# TODO: Use SparseComplexChol. Using GenericComplexChol for now.

import sys
import os
import h5py
import numpy as np

from pyscf import gto, scf, ao2mo
from ueg import UEG
from ipie.config import MPI
from ipie.systems.generic import Generic
#from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.hamiltonians.sparse import Sparse as HamGeneric
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.walkers.walkers_dispatch import UHFWalkersTrial
from ipie.estimators.energy import local_energy
from ipie.propagation.propagator import Propagator
from ipie.utils.mpi import MPIHandler
from ipie.qmc.options import QMCParams
from ipie.qmc.afqmc import AFQMC
from ipie.analysis.extraction import extract_observable

mpi_handler = MPIHandler()
comm = mpi_handler.comm
verbose = 0

if comm.rank == 0:
    print(f'\n# nprocs = {comm.Get_size()}')
    verbose = 1

# AFQMC params.
num_walkers = 10
total_num_walkers = num_walkers * comm.size
num_blocks = 10
num_steps_per_block = 25
timestep = 0.005
seed = 59306159

# Sets the seed in numpy!
np.random.seed(7)

# -----------------------------------------------------------------------------
# Generate UEG integrals.
opts = {
        "nup": 7,
        "ndown": 7,
        "rs": 1.,
        "ecut": 2.5
        }

ueg = UEG(opts, verbose=verbose)
nbasis = ueg.nbasis
nchol = ueg.nchol
nelec = (ueg.nup, ueg.ndown)
nup, ndown = nelec

h1 = ueg.H1[0]
chol = 2. * ueg.chol_vecs.toarray()

# For pyscf.
U = ueg.compute_real_transformation()
h1_8 = U.T.conj() @ h1 @ U
eri_8 = ueg.eri_8() # 8-fold eri
eri_8 = ao2mo.restore(8, eri_8, nbasis)

if verbose: print(np.amax(np.absolute(chol.imag)))

if verbose:
    #print(f'\nfname = {fname}')
    print(f"# nbasis = {nbasis}")
    print(f"# nchol = {nchol}")
    print(f"# nup = {nup}")
    print(f"# ndown = {ndown}")

# -----------------------------------------------------------------------------
# 1. Build out system
system = Generic(nelec=nelec, verbose=verbose)

# -----------------------------------------------------------------------------
# 2. Build Hamiltonian
ham = HamGeneric(
        np.array([h1, h1], dtype=np.complex128), 
        np.array(chol, dtype=np.complex128), 
        0.0, 
        verbose=verbose)

# -----------------------------------------------------------------------------
# 3. Build trial wavefunction
mol = gto.M()
mol.verbose = 5
mol.nelectron = np.sum(nelec)
mol.spin = nup - ndown
mol.max_memory = 60000 # MB
mol.incore_anyway = True

# PW guess.
dm0a = np.zeros(nbasis)
dm0b = np.zeros(nbasis)
dm0a[:nup] = 1
dm0b[:ndown] = 1
dm0 = np.array([np.diag(dm0a), np.diag(dm0b)])

# Random guess.
#dm0a = np.random.random((nbasis, nbasis))
#dm0b = np.random.random((nbasis, nbasis))
#dm0a += dm0a.T.conj()
#dm0b += dm0b.T.conj()
#dm0a *= nup / np.sum(np.trace(dm0a))
#dm0b *= ndown / np.sum(np.trace(dm0b))
#dm0 = np.array([dm0a, dm0b])
#print(np.trace(dm0a))
#print(np.trace(dm0b))

mf = scf.UHF(mol)
#mf.level_shift = 0.5
mf.max_cycle = 5000
mf.get_hcore = lambda *args: h1_8
mf.get_ovlp = lambda *args: np.eye(nbasis)
mf._eri = eri_8
e = mf.kernel(dm0)

Ca, Cb = mf.mo_coeff
psia = Ca[:, :nup]
psib = Cb[:, :ndown]
psi0 = np.zeros((nbasis, np.sum(nelec)), dtype=np.complex128)
psi0[:, :nup] = psia
psi0[:, nup:] = psib
trial = SingleDet(psi0, nelec, nbasis, verbose=verbose)
trial.build()
trial.half_rotate(ham)

trial.calculate_energy(system, ham)
if verbose: print(f"\n# trial.energy = {trial.energy}\n")

assert np.allclose(trial.energy, e)

# Check against RHF solutions of 10.1063/1.5109572
assert np.allclose(np.around(trial.energy, 6), 13.603557) # rs = 1, nbasis = 57

# -----------------------------------------------------------------------------
# 4. Build walkers
walkers = UHFWalkersTrial(trial, psi0, system.nup, system.ndown, ham.nbasis, num_walkers, mpi_handler=mpi_handler, verbose=verbose)
walkers.build(trial)
walkers.ovlp = trial.calc_greens_function(walkers, build_full=True)

# -----------------------------------------------------------------------------
# 5. Build propagator
propagator = Propagator[type(ham)](timestep)
propagator.build(ham, trial, walkers, mpi_handler, verbose)

# -----------------------------------------------------------------------------
# 6. Build AFQMC
params = QMCParams(
            num_walkers = num_walkers,
            total_num_walkers = num_walkers * comm.size,
            num_blocks = num_blocks,
            num_steps_per_block = num_steps_per_block,
            timestep = timestep,
            rng_seed = seed,
        )

afqmc = AFQMC(
            system,
            ham,
            trial,
            walkers,
            propagator,
            params,
            verbose=(verbose and comm.rank == 0),
            )

afqmc.run()

# Extract energy.
qmc_data = extract_observable(afqmc.estimators.filename, "energy")
print()
print(qmc_data.head())
