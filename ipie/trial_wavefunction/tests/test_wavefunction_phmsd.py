import numpy as np
import pytest

from ipie.utils.testing import get_random_phmsd_opt
from ipie.trial_wavefunction.particle_hole import (
        ParticleHoleWicks,
        ParticleHoleWicksBasic,
        ParticleHoleWicksChunked,
        )

@pytest.mark.unit
def test_wicks_basic():
    num_basis = 10
    nalpha, nbeta = (5, 5)
    np.random.seed(7)
    wavefunction, _ = get_random_phmsd_opt(nalpha, nbeta, num_basis, ndet=100)
    trial = ParticleHoleWicksBasic(
            wavefunction,
            (nalpha, nbeta),
            num_basis,
            )
    assert trial.num_elec == (nalpha, nbeta)
    assert trial.num_basis == num_basis
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

@pytest.mark.unit
def test_wicks_opt():
    np.random.seed(7)
    num_basis = 10
    nalpha, nbeta = (5, 5)
    wavefunction, _ = get_random_phmsd_opt(nalpha, nbeta, num_basis, ndet=100)
    trial = ParticleHoleWicks(
            wavefunction,
            (nalpha, nbeta),
            num_basis,
            )
    assert trial.num_elec == (nalpha, nbeta)
    assert trial.num_basis == num_basis
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

@pytest.mark.unit
def test_wicks_opt_chunked():
    np.random.seed(7)
    num_basis = 10
    nalpha, nbeta = (5, 5)
    wavefunction, _ = get_random_phmsd_opt(nalpha, nbeta, num_basis, ndet=100)
    trial = ParticleHoleWicksChunked(
            wavefunction,
            (nalpha, nbeta),
            num_basis,
            num_det_chunks=10,
            )
    assert trial.num_elec == (nalpha, nbeta)
    assert trial.num_basis == num_basis
    assert trial.num_dets == len(trial.occa)
    trial.num_dets = 100
    trial.build()
    num_dets = 1
    for ichunk in range(trial.num_det_chunks):
        for iex, ex_list in enumerate(trial.cre_ex_a_chunk[ichunk]):
            num_dets += len(ex_list)
    assert num_dets == trial.num_dets
    trial.num_dets = 200
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
