import numpy as np
import pytest

from ipie.config import MPI
from ipie.estimators.local_energy import variational_energy_ortho_det
from ipie.lib.libci import _HAVE_OPT_CI_CODE
from ipie.trial_wavefunction.particle_hole import (
    ParticleHole,
    ParticleHoleNonChunked,
    ParticleHoleSlow,
)
from ipie.utils.testing import build_random_ci_wfn, get_random_phmsd_opt, get_random_sys_ham


@pytest.mark.unit
def test_wicks_slow():
    nbasis = 10
    nalpha, nbeta = (5, 5)
    np.random.seed(7)
    wavefunction, _ = get_random_phmsd_opt(nalpha, nbeta, nbasis, ndet=100)
    trial = ParticleHoleSlow(
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
    trial.num_dets_for_props = 10
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
    trial = ParticleHoleNonChunked(
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
    trial = ParticleHole(
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


@pytest.mark.libci
@pytest.mark.skipif(not _HAVE_OPT_CI_CODE, reason="lib.libci not found.")
@pytest.mark.parametrize(
    "nalpha,nbeta,nbasis", ((4, 4, 10), (4, 7, 12), (36, 36, 47), (64, 65, 72), (8, 9, 109))
)
def test_opt_opdm(nalpha, nbeta, nbasis):
    wavefunction = build_random_ci_wfn(nbasis, (nalpha, nbeta), 100, cmplx_coeffs=False)
    trial = ParticleHole(
        wavefunction,
        (nalpha, nbeta),
        nbasis,
        verbose=False,
        num_dets_for_props=len(wavefunction[0]),
    )
    ref = trial.compute_1rdm(nbasis)
    assert np.allclose(trial.G, ref)
    wavefunction = build_random_ci_wfn(nbasis, (nalpha, nbeta), 100, cmplx_coeffs=True)
    trial = ParticleHole(
        wavefunction,
        (nalpha, nbeta),
        nbasis,
        verbose=False,
        num_dets_for_props=(len(wavefunction[0])),
    )
    ref = trial.compute_1rdm(nbasis)
    assert np.allclose(trial.G[0], ref[0].T)
    assert np.allclose(trial.G[1], ref[1].T)


@pytest.mark.libci
@pytest.mark.skipif(not _HAVE_OPT_CI_CODE, reason="lib.libci not found.")
@pytest.mark.parametrize("nmelting,nalpha,nbeta,nbasis", ((18, 64, 65, 72), (3, 8, 9, 109)))
def test_opt_opdm_melting(nmelting, nalpha, nbeta, nbasis):
    wavefunction = build_random_ci_wfn(
        nbasis - nmelting, (nalpha - nmelting, nbeta - nmelting), 100, cmplx_coeffs=False
    )
    trial = ParticleHole(
        wavefunction,
        (nalpha, nbeta),
        nbasis,
        verbose=True,
        num_dets_for_props=len(wavefunction[0]),
    )
    ref = trial.compute_1rdm(nbasis)
    assert np.allclose(trial.G, ref)
    wavefunction = build_random_ci_wfn(
        nbasis - nmelting, (nalpha - nmelting, nbeta - nmelting), 100, cmplx_coeffs=True
    )
    trial = ParticleHole(
        wavefunction,
        (nalpha, nbeta),
        nbasis,
        verbose=False,
        num_dets_for_props=(len(wavefunction[0])),
    )
    ref = trial.compute_1rdm(nbasis)
    assert np.allclose(trial.G[0], ref[0].T)
    assert np.allclose(trial.G[1], ref[1].T)


@pytest.mark.libci
@pytest.mark.skipif(not _HAVE_OPT_CI_CODE, reason="lib.libci not found.")
@pytest.mark.parametrize("nalpha,nbeta,nbasis", ((10, 10, 44), (8, 9, 32)))
def test_opt_opdm_variational_energy(nalpha, nbeta, nbasis):
    wavefunction = build_random_ci_wfn(nbasis, (nalpha, nbeta), 100, cmplx_coeffs=False)

    trial = ParticleHole(
        wavefunction,
        (nalpha, nbeta),
        nbasis,
        verbose=True,
        num_dets_for_props=len(wavefunction[0]),
    )
    sys, ham = get_random_sys_ham(nalpha, nbeta, nbasis, 4 * nbasis, symmetrize=True)
    ref = variational_energy_ortho_det(sys, ham, trial.spin_occs, trial.coeffs)
    test = trial.calculate_energy(sys, ham)
    assert np.allclose(ref, test)
    wavefunction = build_random_ci_wfn(nbasis, (nalpha, nbeta), 100, cmplx_coeffs=True)
    trial = ParticleHole(
        wavefunction,
        (nalpha, nbeta),
        nbasis,
        verbose=False,
        num_dets_for_props=(len(wavefunction[0])),
    )
    ref = variational_energy_ortho_det(sys, ham, trial.spin_occs, trial.coeffs)
    test = trial.calculate_energy(sys, ham)
    assert np.allclose(ref, test)
