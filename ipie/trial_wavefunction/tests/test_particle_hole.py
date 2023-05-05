from mpi4py import MPI
import numpy as np
import pytest

from ipie.utils.testing import get_random_phmsd_opt, get_random_sys_ham
from ipie.trial_wavefunction.particle_hole import (
    ParticleHoleWicks,
    ParticleHoleWicksSlow,
    ParticleHoleWicksNonChunked,
)


@pytest.mark.unit
def test_wicks_slow():
    nbasis = 10
    nalpha, nbeta = (5, 5)
    np.random.seed(7)
    wavefunction, _ = get_random_phmsd_opt(nalpha, nbeta, nbasis, ndet=100)
    trial = ParticleHoleWicksSlow(
        wavefunction,
        (nalpha, nbeta),
        nbasis,
    )
    assert trial.nelec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.num_dets == len(trial.occa)
    trial.build()
    assert len(trial.cre_a) == trial.num_dets
    assert len(trial.cre_b) == trial.num_dets
    assert len(trial.phase_a) == trial.num_dets
    assert len(trial.phase_b) == trial.num_dets
    trial.num_dets = 10
    trial.build()
    assert len(trial.cre_a) == trial.num_dets
    assert len(trial.cre_b) == trial.num_dets
    assert len(trial.phase_a) == trial.num_dets
    assert len(trial.phase_b) == trial.num_dets
    naux = 10 * nbasis
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    comm = MPI.COMM_WORLD
    trial.half_rotate(ham, comm=comm)
    assert trial._rchola.shape == (naux, nbasis * nalpha)
    assert trial._rcholb.shape == (naux, nbasis * nbeta)
    assert trial._rH1a.shape == (nalpha, nbasis)
    assert trial._rH1b.shape == (nbeta, nbasis)
    assert trial._rchola_act.shape == (naux, nbasis * trial.nact)
    assert trial._rcholb_act.shape == (naux, nbasis * trial.nact)


@pytest.mark.unit
def test_wicks_opt():
    np.random.seed(7)
    nbasis = 10
    nalpha, nbeta = (5, 5)
    wavefunction, _ = get_random_phmsd_opt(nalpha, nbeta, nbasis, ndet=100)
    trial = ParticleHoleWicksNonChunked(
        wavefunction,
        (nalpha, nbeta),
        nbasis,
    )
    assert trial.nelec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.num_dets == len(trial.occa)
    trial.num_dets = 100
    trial.build()
    num_dets = 1
    for iex, ex_list in enumerate(trial.cre_ex_a):
        num_dets += len(ex_list)
    assert num_dets == trial.num_dets
    num_dets = 1
    for iex, ex_list in enumerate(trial.cre_ex_b):
        num_dets += len(ex_list)
    assert num_dets == trial.num_dets
    num_dets = 1
    for iex, ex_list in enumerate(trial.anh_ex_b):
        num_dets += len(ex_list)
    assert num_dets == trial.num_dets
    num_dets = 1
    for iex, ex_list in enumerate(trial.anh_ex_a):
        num_dets += len(ex_list)
    assert num_dets == trial.num_dets
    assert len(trial.phase_a) == trial.num_dets
    assert len(trial.phase_b) == trial.num_dets
    naux = 10 * nbasis
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    comm = MPI.COMM_WORLD
    trial.half_rotate(ham, comm=comm)
    assert trial._rchola.shape == (naux, nbasis * nalpha)
    assert trial._rcholb.shape == (naux, nbasis * nbeta)
    assert trial._rH1a.shape == (nalpha, nbasis)
    assert trial._rH1b.shape == (nbeta, nbasis)
    assert trial._rchola_act.shape == (naux, nbasis * trial.nact)
    assert trial._rcholb_act.shape == (naux, nbasis * trial.nact)


@pytest.mark.unit
def test_wicks_opt_chunked():
    np.random.seed(7)
    nbasis = 10
    nalpha, nbeta = (5, 5)
    wavefunction, _ = get_random_phmsd_opt(nalpha, nbeta, nbasis, ndet=100)
    trial = ParticleHoleWicks(
        wavefunction,
        (nalpha, nbeta),
        nbasis,
        num_det_chunks=10,
    )
    assert trial.nelec == (nalpha, nbeta)
    assert trial.nbasis == nbasis
    assert trial.num_dets == len(trial.occa)
    trial.num_dets = 100
    trial.build()
    num_dets = 1
    for ichunk in range(trial.num_det_chunks):
        for iex, ex_list in enumerate(trial.cre_ex_a_chunk[ichunk]):
            num_dets += len(ex_list)
    assert num_dets == trial.num_dets
    trial.num_dets = 20
    trial.build()
    num_dets = 1
    for ichunk in range(trial.num_det_chunks):
        for iex, ex_list in enumerate(trial.cre_ex_a_chunk[ichunk]):
            num_dets += len(ex_list)
    assert num_dets == trial.num_dets
    num_dets = 1
    for ichunk in range(trial.num_det_chunks):
        for iex, ex_list in enumerate(trial.anh_ex_a_chunk[ichunk]):
            num_dets += len(ex_list)
    assert num_dets == trial.num_dets
    num_dets = 1
    for ichunk in range(trial.num_det_chunks):
        for iex, ex_list in enumerate(trial.cre_ex_b_chunk[ichunk]):
            num_dets += len(ex_list)
    assert num_dets == trial.num_dets
    num_dets = 1
    for ichunk in range(trial.num_det_chunks):
        for iex, ex_list in enumerate(trial.cre_ex_b_chunk[ichunk]):
            num_dets += len(ex_list)
    assert num_dets == trial.num_dets
    naux = 10 * nbasis
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, naux)
    comm = MPI.COMM_WORLD
    trial.half_rotate(ham, comm=comm)
    assert trial._rchola.shape == (naux, nbasis * nalpha)
    assert trial._rcholb.shape == (naux, nbasis * nbeta)
    assert trial._rH1a.shape == (nalpha, nbasis)
    assert trial._rH1b.shape == (nbeta, nbasis)
    assert trial._rchola_act.shape == (naux, nbasis * trial.nact)
    assert trial._rcholb_act.shape == (naux, nbasis * trial.nact)
