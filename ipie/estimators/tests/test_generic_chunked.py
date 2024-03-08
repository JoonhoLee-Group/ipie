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

import numpy
import pytest

from ipie.config import MPI
from ipie.estimators.local_energy_sd import local_energy_single_det_batch
from ipie.estimators.local_energy_sd_chunked import local_energy_single_det_uhf_batch_chunked
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.propagation.phaseless_generic import PhaselessGenericChunked
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.misc import dotdict
from ipie.utils.mpi import get_shared_array, MPIHandler
from ipie.utils.pack_numba import pack_cholesky
from ipie.utils.testing import generate_hamiltonian, get_random_nomsd
from ipie.walkers.walkers_dispatch import UHFWalkersTrial

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numpy.random.seed(7)
skip = comm.size == 1


@pytest.mark.unit
@pytest.mark.skipif(skip, reason="Test should be run on multiple cores.")
def test_generic_chunked():
    nwalkers = 50
    nsteps = 20
    numpy.random.seed(7)
    nmo = 24
    nelec = (4, 2)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)

    h1e = comm.bcast(h1e)
    chol = comm.bcast(chol)
    enuc = comm.bcast(enuc)
    eri = comm.bcast(eri)

    chol = chol.reshape((-1, nmo * nmo)).T.copy()

    nchol = chol.shape[-1]
    chol = chol.reshape((nmo, nmo, nchol))

    idx = numpy.triu_indices(nmo)
    cp_shape = (nmo * (nmo + 1) // 2, chol.shape[-1])
    # chol_packed = numpy.zeros(cp_shape, dtype = chol.dtype)
    chol_packed = get_shared_array(comm, cp_shape, chol.dtype)

    if comm.rank == 0:
        pack_cholesky(idx[0], idx[1], chol_packed, chol)

    chol = chol.reshape((nmo * nmo, nchol))

    system = Generic(nelec=nelec)
    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]), chol=chol, ecore=enuc)
    _, wfn = get_random_nomsd(system.nup, system.ndown, hamiltonian.nbasis, ndet=1, cplx=False)
    trial = SingleDet(wfn[0], nelec, nmo)
    trial.half_rotate(hamiltonian)

    trial.calculate_energy(system, hamiltonian)

    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})

    mpi_handler = MPIHandler(nmembers=3, verbose=(rank == 0))
    if comm.rank == 0:
        print("# Chunking hamiltonian.")
    hamiltonian.chunk(mpi_handler)
    if comm.rank == 0:
        print("# Chunking trial.")
    trial.chunk(mpi_handler)

    prop = PhaselessGenericChunked(time_step=qmc["dt"])
    prop.build(hamiltonian, trial, mpi_handler=mpi_handler)

    init_walker = numpy.hstack([trial.psi0a, trial.psi0b])
    walkers = UHFWalkersTrial(
        trial, init_walker, system.nup, system.ndown, hamiltonian.nbasis, nwalkers, mpi_handler=mpi_handler
    )
    walkers.build(trial)

    for i in range(nsteps):
        prop.propagate_walkers(walkers, hamiltonian, trial, trial.energy)
        walkers.reortho()

    energies = local_energy_single_det_batch(system, hamiltonian, walkers, trial)

    energies_chunked = local_energy_single_det_uhf_batch_chunked(system, hamiltonian, walkers, trial)

    assert numpy.allclose(energies, energies_chunked)


if __name__ == "__main__":
    test_generic_chunked()
