import numpy
import pytest
import scipy.linalg

from ipie.estimators.energy import local_energy
from ipie.hamiltonians.hubbard import Hubbard, HubbardSpinOrbital
from ipie.propagation.hirsch_base import HirschBase, construct_one_body_propagator
from ipie.propagation.hubbard_generic import HubbardSingleSite, _numba_ghf_kernels
from ipie.qmc.afqmc import AFQMC
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.mpi import MPIHandler
from ipie.walkers.ghf_walkers import GHFWalkers

pytestmark = pytest.mark.skipif(_numba_ghf_kernels() is None, reason="numba not available")

NBASIS = 6
NELEC = (3, 2)
NWALKERS = 5


def random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def random_hermitian(rng, n):
    a = random_complex(rng, (n, n))
    return 0.5 * (a + a.conj().T)


def repulsive_hubbard(rng):
    t = random_hermitian(rng, NBASIS)
    return Hubbard(numpy.array([t, t.conj()]), U=4.0)


def attractive_spin_orbital(rng):
    # Spin-flip blocks included: only the total particle number is conserved.
    return HubbardSpinOrbital(random_hermitian(rng, 2 * NBASIS), U=-4.0, ecore=-1.5)


def build_case(make_ham, seed):
    rng = numpy.random.default_rng(seed)
    ham = make_ham(rng)
    nocc = sum(NELEC)
    psi0 = numpy.linalg.qr(random_complex(rng, (2 * NBASIS, nocc)))[0]
    phi = psi0[None] + 0.3 * random_complex(rng, (NWALKERS, 2 * NBASIS, nocc))
    trial = SingleDetGHF(psi0, NELEC, NBASIS)
    walkers = GHFWalkers(phi[0], NELEC[0], NELEC[1], NBASIS, NWALKERS, MPIHandler())
    walkers.phi = phi.copy()
    walkers.build(trial)
    walkers.weight = rng.uniform(0.5, 1.5, NWALKERS)
    walkers.weight[1] = 0.0
    prop = HubbardSingleSite(0.05)
    prop.build(ham, trial, walkers, walkers.mpi_handler)
    fields = rng.random((NBASIS, NWALKERS))
    return ham, trial, walkers, prop, fields


def get_state(walkers):
    return [x.copy() for x in (walkers.phi, walkers.inv_ovlp, walkers.ovlp, walkers.weight)]


def set_state(walkers, state):
    walkers.phi, walkers.inv_ovlp, walkers.ovlp, walkers.weight = [x.copy() for x in state]


def assert_same_state(walkers, ref):
    for new, old in zip(get_state(walkers), ref):
        numpy.testing.assert_allclose(new, old, rtol=1e-9, atol=1e-12)


def assert_consistent(walkers, trial):
    ovlp = numpy.einsum("mi,wmj->wij", trial.psi0.conj(), walkers.phi)
    numpy.testing.assert_allclose(walkers.inv_ovlp, numpy.linalg.inv(ovlp), rtol=1e-9, atol=1e-10)
    numpy.testing.assert_allclose(walkers.ovlp, trial.calc_overlap(walkers), rtol=1e-9, atol=1e-12)


CASES = [repulsive_hubbard, attractive_spin_orbital]


@pytest.mark.unit
@pytest.mark.parametrize("make_ham", CASES)
def test_numba_ghf_two_body_smoke(make_ham):
    ham, trial, walkers, prop, fields = build_case(make_ham, 1)
    assert prop._choose_path(walkers, ham) == "ghf_numba"
    prop._numba_ghf_two_body(walkers, ham, trial, fields)
    assert numpy.all(numpy.isfinite(walkers.phi))
    assert numpy.all(numpy.isfinite(walkers.weight))
    assert walkers.weight[1] == 0.0
    assert_consistent(walkers, trial)


@pytest.mark.unit
@pytest.mark.parametrize("make_ham", CASES)
def test_numba_ghf_two_body_matches_einsum(make_ham):
    ham, trial, walkers, prop, fields = build_case(make_ham, 2)
    start = get_state(walkers)
    prop._ghf_two_body(walkers, ham, trial, fields)
    ref = get_state(walkers)
    set_state(walkers, start)
    prop._numba_ghf_two_body(walkers, ham, trial, fields)
    assert_same_state(walkers, ref)


@pytest.mark.unit
@pytest.mark.parametrize("make_ham", CASES)
def test_einsum_ghf_two_body_tracks_overlap(make_ham):
    # The charge decoupling has aux_wfac != 1; the tracked overlap must still be <psi_T|phi>.
    ham, trial, walkers, prop, fields = build_case(make_ham, 3)
    prop._ghf_two_body(walkers, ham, trial, fields)
    assert_consistent(walkers, trial)


@pytest.mark.unit
@pytest.mark.parametrize("make_ham", CASES)
def test_numba_ghf_kinetic_matches_reference(make_ham):
    ham, trial, walkers, prop, _ = build_case(make_ham, 4)
    start = get_state(walkers)
    HirschBase.kinetic_importance_sampling(prop, walkers, trial)
    ref = get_state(walkers)
    set_state(walkers, start)
    prop.kinetic_importance_sampling(walkers, trial)
    assert_same_state(walkers, ref)
    assert_consistent(walkers, trial)


@pytest.mark.unit
def test_hirsch_decoupling_follows_sign_of_u():
    rng = numpy.random.default_rng(5)
    for make_ham, spin in ((repulsive_hubbard, True), (attractive_spin_orbital, False)):
        ham = make_ham(rng)
        prop = HubbardSingleSite(0.05)
        prop.build(ham)
        assert prop.spin_decomp is spin
        assert numpy.isfinite(prop.gamma) and abs(numpy.imag(prop.gamma)) < 1e-14


@pytest.mark.unit
def test_spin_orbital_block_diagonal_matches_hubbard():
    rng = numpy.random.default_rng(6)
    t_up = random_hermitian(rng, NBASIS)
    t_dn = random_hermitian(rng, NBASIS)
    ham = Hubbard(numpy.array([t_up, t_dn]), U=3.0, ecore=0.7)
    ham_so = HubbardSpinOrbital(scipy.linalg.block_diag(t_up, t_dn), U=3.0, ecore=0.7)
    _, trial, walkers, _, _ = build_case(lambda r: ham, 7)

    exp_h1 = construct_one_body_propagator(ham, 0.05)
    numpy.testing.assert_allclose(
        construct_one_body_propagator(ham_so, 0.05), scipy.linalg.block_diag(*exp_h1), atol=1e-13
    )
    trial.calc_greens_function(walkers, build_full=True)
    system = Generic(NELEC)
    numpy.testing.assert_allclose(
        local_energy(system, ham_so, walkers, trial), local_energy(system, ham, walkers, trial), atol=1e-11
    )
    trial.calculate_energy(system, ham)
    e_ref = trial.energy
    trial.calculate_energy(system, ham_so)
    numpy.testing.assert_allclose(trial.energy, e_ref, atol=1e-11)


@pytest.mark.unit
def test_spin_orbital_afqmc_smoke(tmp_path):
    rng = numpy.random.default_rng(8)
    ham = attractive_spin_orbital(rng)
    psi0 = numpy.linalg.eigh(ham.H1)[1][:, : sum(NELEC)]
    trial = SingleDetGHF(psi0, NELEC, NBASIS)
    trial.calculate_energy(Generic(NELEC), ham)
    afqmc = AFQMC.build(
        NELEC,
        ham,
        trial,
        num_walkers=8,
        seed=9,
        num_steps_per_block=5,
        num_blocks=3,
        timestep=0.02,
        pop_control_method="comb",
        verbose=False,
    )
    afqmc.run(estimator_filename=str(tmp_path / "estimates.h5"), verbose=False)
    assert afqmc.propagator._choose_path(afqmc.walkers, ham) == "ghf_numba"
    assert numpy.all(numpy.isfinite(afqmc.walkers.weight))
    assert numpy.all(numpy.isfinite(afqmc.walkers.ovlp))
