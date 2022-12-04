
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
from mpi4py import MPI

from ipie.estimators.generic import local_energy_cholesky_opt
from ipie.estimators.local_energy_sd import (local_energy_single_det_batch,
                                             local_energy_single_det_rhf_batch,
                                             local_energy_single_det_uhf_batch)
from ipie.estimators.local_energy_sd_chunked import \
    local_energy_single_det_uhf_batch_chunked
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.propagation.continuous import Continuous
from ipie.propagation.force_bias import (
    construct_force_bias_batch_single_det,
    construct_force_bias_batch_single_det_chunked)
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.utils.misc import dotdict, is_cupy
from ipie.utils.mpi import MPIHandler, get_shared_array, have_shared_mem
from ipie.utils.pack import pack_cholesky
from ipie.utils.testing import generate_hamiltonian, get_random_nomsd
from ipie.walkers.single_det_batch import SingleDetWalkerBatch

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

numpy.random.seed(7)
skip = comm.size == 1


@pytest.mark.unit
@pytest.mark.skipif(skip, reason="Test should be run on multiple cores.")
def test_generic_propagation_chunked():
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
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]), chol=chol, chol_packed=chol_packed, ecore=enuc
    )
    wfn = get_random_nomsd(system.nup, system.ndown, ham.nbasis, ndet=1, cplx=False)
    trial = MultiSlater(system, ham, wfn)
    trial.half_rotate(system, ham)

    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.calculate_energy(system, ham)

    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    options = {"hybrid": True}
    prop = Continuous(system, ham, trial, qmc, options=options)

    mpi_handler = MPIHandler(comm, options={"nmembers": 3}, verbose=(rank == 0))
    ham.chunk(mpi_handler)
    trial.chunk(mpi_handler)

    walker_batch = SingleDetWalkerBatch(
        system, ham, trial, nwalkers, mpi_handler=mpi_handler
    )
    for i in range(nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, trial.energy)
        walker_batch.reortho()

    vfb = construct_force_bias_batch_single_det(ham, walker_batch, trial)
    vfb_chunked = construct_force_bias_batch_single_det_chunked(
        ham, walker_batch, trial, mpi_handler
    )

    assert numpy.allclose(vfb, vfb_chunked)
    xshifted = numpy.random.normal(0.0, 1.0, ham.nchol * walker_batch.nwalkers).reshape(
        walker_batch.nwalkers, ham.nchol
    )
    VHS_chunked = prop.propagator.construct_VHS_batch_chunked(
        ham, xshifted.T.copy(), walker_batch.mpi_handler
    )
    VHS = prop.propagator.construct_VHS_batch(ham, xshifted.T.copy())
    assert numpy.allclose(VHS, VHS_chunked)


if __name__ == "__main__":
    test_generic_propagation_chunked()
