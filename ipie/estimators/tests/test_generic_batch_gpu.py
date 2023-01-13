
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

from math import ceil
import numpy
import pytest
import sys

try:
    import cupy
except:
    no_gpu = True


from ipie.estimators.local_energy_sd import (local_energy_single_det_batch,
                                             local_energy_single_det_batch_gpu)
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.propagation.continuous import Continuous
from ipie.systems.generic import Generic
from ipie.utils.misc import dotdict
from ipie.utils.pack import pack_cholesky
from ipie.utils.testing import (generate_hamiltonian,
                                get_random_phmsd, shaped_normal)
from ipie.walkers.single_det_batch import SingleDetWalkerBatch
from ipie.trial_wavefunction.single_det import SingleDet


@pytest.mark.gpu
def test_exchange_kernel_reduction():
    import cupy
    nchol = 101
    nocc = 31
    nwalk = 7
    chunk_size = 10
    T = cupy.array(shaped_normal((nchol, nocc, nwalk, nocc), cmplx=True))
    buff = T.copy().reshape((nchol, -1))
    exx = cupy.einsum("xjwi,xiwj->w", T, T)
    exx_test = cupy.zeros_like(exx)
    from ipie.estimators.kernels.gpu import exchange as kernels
    kernels.exchange_reduction(T, exx_test)
    assert numpy.allclose(exx_test, exx)
    nchol_left = nchol
    exx_test = cupy.zeros_like(exx)
    exx_test_2 = cupy.zeros_like(exx)
    for i in range(ceil(nchol / 10)):
        nchol_chunk = min(chunk_size, nchol_left)
        chol_sls = slice(i * chunk_size, i * chunk_size + nchol_chunk)
        size = nwalk * nchol_chunk * nocc * nocc
        # alpha-alpha
        Txij = buff[chol_sls].reshape((nchol_chunk, nocc, nwalk, nocc))
        kernels.exchange_reduction(Txij, exx_test)
        nchol_left -= chunk_size
    assert numpy.allclose(exx_test, exx)


@pytest.mark.gpu
def test_local_energy_single_det_batch():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 25
    from ipie.utils.backend import arraylib as xp
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)

    chol = chol.reshape((-1, nmo * nmo)).T.copy()

    nchol = chol.shape[-1]
    chol = chol.reshape((nmo, nmo, nchol))

    idx = numpy.triu_indices(nmo)
    cp_shape = (nmo * (nmo + 1) // 2, chol.shape[-1])
    chol_packed = numpy.zeros(cp_shape, dtype=chol.dtype)
    pack_cholesky(idx[0], idx[1], chol_packed, chol)
    chol = chol.reshape((nmo * nmo, nchol))

    system = Generic(nelec=nelec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]), chol=chol, chol_packed=chol_packed, ecore=0
    )
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(
        system.nup, system.ndown, ham.nbasis, ndet=1, init=True
    )
    Id = numpy.eye(ham.nbasis)
    wfn = numpy.hstack([Id[:,:system.nup], Id[:,:system.ndown]])
    trial = SingleDet(wfn, system.nelec, ham.nbasis, init=init)
    trial.half_rotate(system, ham)
    # trial.calculate_energy(system, ham)

    numpy.random.seed(7)

    options = {"hybrid": True}
    qmc = dotdict({"dt": 0.005, "nstblz": 5, "batched": True, "nwalkers": nwalkers})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)

    prop.cast_to_cupy()
    ham.cast_to_cupy()
    trial.cast_to_cupy()
    walker_batch.cast_to_cupy()

    for i in range(nsteps):
        prop.propagate_walker_batch(walker_batch, system, ham, trial, trial.energy)
        walker_batch.reortho()

    energies_einsum = local_energy_single_det_batch_gpu(
        system, ham, walker_batch, trial
    )

    energies_einsum_chunks = local_energy_single_det_batch_gpu(
        system, ham, walker_batch, trial, max_mem=1e-6,
    )
    from ipie.estimators.local_energy_sd import local_energy_single_det_batch_gpu_old

    energies_einsum_old = local_energy_single_det_batch_gpu_old(
        system, ham, walker_batch, trial
    )
    walker_batch.Ghalfa = cupy.asnumpy(walker_batch.Ghalfa)
    walker_batch.Ghalfb = cupy.asnumpy(walker_batch.Ghalfb)
    trial._rchola = cupy.asnumpy(trial._rchola)
    trial._rcholb = cupy.asnumpy(trial._rcholb)
    trial._rH1a = cupy.asnumpy(trial._rH1a)
    trial._rH1b = cupy.asnumpy(trial._rH1b)
    energies = local_energy_single_det_batch(system, ham, walker_batch, trial)

    assert numpy.allclose(energies, energies_einsum_old)
    assert numpy.allclose(energies, energies_einsum)
    assert numpy.allclose(energies, energies_einsum_chunks)

if __name__ == "__main__":
    test_local_energy_single_det_batch()
