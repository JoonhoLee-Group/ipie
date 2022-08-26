from math import ceil
import numpy
import pytest
import sys

try:
    import cupy
    # from ipie.config import config, purge_ipie_modules
    # config.update_option('use_gpu', True)
except:
    no_gpu = True


from ipie.estimators.local_energy_sd import (local_energy_single_det_batch,
                                             local_energy_single_det_batch_gpu)
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.legacy.estimators.local_energy import local_energy_generic_cholesky_opt
from ipie.legacy.walkers.multi_det import MultiDetWalker
from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.propagation.continuous import Continuous
from ipie.propagation.force_bias import construct_force_bias_batch
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.utils.misc import dotdict
from ipie.utils.pack import pack_cholesky
from ipie.utils.testing import (generate_hamiltonian, get_random_nomsd,
                                get_random_phmsd, shaped_normal)
from ipie.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from ipie.walkers.single_det_batch import SingleDetWalkerBatch


@pytest.mark.gpu
def test_exchange_kernel_reduction(gpu_env):
    import cupy
    # from ipie.config import config, purge_ipie_modules
    # config.update_option('use_gpu', True)
    from ipie.utils.backend import arraylib as xp
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
def test_local_energy_single_det_batch(gpu_env):
    numpy.random.seed(7)
    nmo = 10
    nelec = (5, 5)
    nwalkers = 10
    nsteps = 25
    from ipie.utils.backend import arraylib as xp
    print(xp)
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
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.half_rotate(system, ham)
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    trial.calculate_energy(system, ham)

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
