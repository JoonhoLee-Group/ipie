import numpy
import pytest

from ipie.estimators.energy import local_energy
from ipie.hamiltonians.hubbard import Hubbard
from ipie.propagation.hubbard_generic import HubbardSingleSite
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.mpi import MPIHandler
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.walkers.uhf_walkers import UHFWalkers


def random_complex(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def build_random_ghf_case(nbasis=4, nelec=(2, 1), nwalkers=3, seed=7):
    rng = numpy.random.default_rng(seed)
    nocc = sum(nelec)
    psi0 = random_complex(rng, (2 * nbasis, nocc))
    phi = random_complex(rng, (nwalkers, 2 * nbasis, nocc))
    trial = SingleDetGHF(psi0, nelec, nbasis)
    walkers = GHFWalkers(phi[0], nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers.phi = phi.copy()
    walkers.build(trial)
    return trial, walkers


@pytest.mark.unit
def test_ghf_inverse_overlap_matches_direct_inverse():
    trial, walkers = build_random_ghf_case()

    ovlp = numpy.einsum("mi,wmj->wij", trial.psi0.conj(), walkers.phi)
    numpy.testing.assert_allclose(walkers.inv_ovlp, numpy.linalg.inv(ovlp), atol=1e-12)


@pytest.mark.unit
def test_ghf_hubbard_overlap_ratio_matches_direct_determinant():
    trial, walkers = build_random_ghf_case(seed=8)
    h1e = numpy.zeros((2, walkers.nbasis, walkers.nbasis))
    ham = Hubbard(h1e, U=4.0)
    prop = HubbardSingleSite(0.01)
    prop.build(ham, trial, walkers, walkers.mpi_handler)

    site = 2
    up = site
    down = site + walkers.nbasis
    guu, gud, gdu, gdd = prop._ghf_site_greens_block(walkers, trial, site)
    old_ovlp = numpy.linalg.det(numpy.einsum("mi,wmj->wij", trial.psi0.conj(), walkers.phi))

    for field in (0, 1):
        delta_up = prop.delta[field, 0]
        delta_down = prop.delta[field, 1]
        ratio = (
            (1.0 + delta_up * guu) * (1.0 + delta_down * gdd)
            - delta_up * delta_down * gud * gdu
        )

        phi_new = walkers.phi.copy()
        phi_new[:, up, :] *= prop.auxf[field, 0]
        phi_new[:, down, :] *= prop.auxf[field, 1]
        new_ovlp = numpy.linalg.det(numpy.einsum("mi,wmj->wij", trial.psi0.conj(), phi_new))

        numpy.testing.assert_allclose(ratio, new_ovlp / old_ovlp, atol=1e-10)


@pytest.mark.unit
def test_ghf_hubbard_inverse_update_matches_direct_recompute():
    trial, walkers = build_random_ghf_case(seed=9)
    h1e = numpy.zeros((2, walkers.nbasis, walkers.nbasis))
    ham = Hubbard(h1e, U=4.0)
    prop = HubbardSingleSite(0.01)
    prop.build(ham, trial, walkers, walkers.mpi_handler)

    random_fields = numpy.full((walkers.nbasis, walkers.nwalkers), 0.37)
    prop._ghf_two_body(walkers, ham, trial, random_fields)

    ovlp = numpy.einsum("mi,wmj->wij", trial.psi0.conj(), walkers.phi)
    numpy.testing.assert_allclose(walkers.inv_ovlp, numpy.linalg.inv(ovlp), atol=1e-10)
    numpy.testing.assert_allclose(walkers.ovlp, trial.calc_overlap(walkers), atol=1e-10)


@pytest.mark.unit
def test_ghf_hubbard_full_propagation_smoke():
    numpy.random.seed(12)
    trial, walkers = build_random_ghf_case(seed=12)
    h1e = numpy.zeros((2, walkers.nbasis, walkers.nbasis))
    ham = Hubbard(h1e, U=4.0)
    prop = HubbardSingleSite(0.01)
    prop.build(ham, trial, walkers, walkers.mpi_handler)

    prop.propagate_walkers(walkers, ham, trial, eshift=0.0)

    ovlp = numpy.einsum("mi,wmj->wij", trial.psi0.conj(), walkers.phi)
    numpy.testing.assert_allclose(walkers.inv_ovlp, numpy.linalg.inv(ovlp), atol=1e-10)
    numpy.testing.assert_allclose(walkers.ovlp, trial.calc_overlap(walkers), atol=1e-10)
    assert numpy.all(numpy.isfinite(walkers.weight))


@pytest.mark.unit
def test_ghf_hubbard_block_diagonal_matches_uhf_two_body():
    rng = numpy.random.default_rng(10)
    nbasis = 5
    nelec = (2, 2)
    nwalkers = 4
    h1e = numpy.zeros((2, nbasis, nbasis))
    ham = Hubbard(h1e, U=3.0)
    random_fields = numpy.full((nbasis, nwalkers), 0.25)

    psi_uhf = random_complex(rng, (nbasis, sum(nelec)))
    phi_uhf = random_complex(rng, (nbasis, sum(nelec)))
    trial_uhf = SingleDet(psi_uhf, nelec, nbasis)
    walkers_uhf = UHFWalkers(phi_uhf, nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers_uhf.build(trial_uhf)

    psi_ghf = numpy.zeros((2 * nbasis, sum(nelec)), dtype=numpy.complex128)
    psi_ghf[:nbasis, : nelec[0]] = trial_uhf.psi0a
    psi_ghf[nbasis:, nelec[0] :] = trial_uhf.psi0b
    phi_ghf = numpy.zeros_like(psi_ghf)
    phi_ghf[:nbasis, : nelec[0]] = phi_uhf[:, : nelec[0]]
    phi_ghf[nbasis:, nelec[0] :] = phi_uhf[:, nelec[0] :]
    trial_ghf = SingleDetGHF(psi_ghf, nelec, nbasis)
    walkers_ghf = GHFWalkers(phi_ghf, nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers_ghf.build(trial_ghf)

    prop_uhf = HubbardSingleSite(0.01)
    prop_uhf.build(ham, trial_uhf, walkers_uhf, walkers_uhf.mpi_handler)
    prop_ghf = HubbardSingleSite(0.01)
    prop_ghf.build(ham, trial_ghf, walkers_ghf, walkers_ghf.mpi_handler)

    prop_uhf._einsum_two_body(walkers_uhf, ham, trial_uhf, random_fields)
    prop_ghf._ghf_two_body(walkers_ghf, ham, trial_ghf, random_fields)

    numpy.testing.assert_allclose(walkers_uhf.weight, walkers_ghf.weight, atol=1e-12)
    numpy.testing.assert_allclose(walkers_uhf.ovlp, walkers_ghf.ovlp, atol=1e-10)
    numpy.testing.assert_allclose(
        walkers_uhf.phia, walkers_ghf.phi[:, :nbasis, : nelec[0]], atol=1e-12
    )
    numpy.testing.assert_allclose(
        walkers_uhf.phib, walkers_ghf.phi[:, nbasis:, nelec[0] :], atol=1e-12
    )


@pytest.mark.unit
def test_ghf_from_uhf_calculation_matches_uhf_with_same_fields():
    rng = numpy.random.default_rng(13)
    nbasis = 6
    nelec = (3, 2)
    nwalkers = 5
    nsteps = 4
    timestep = 0.01

    h = random_complex(rng, (nbasis, nbasis))
    h = h + h.conj().T
    ham = Hubbard(numpy.array([h, h]), U=3.0)
    system = Generic(nelec=nelec)

    psi_uhf = random_complex(rng, (nbasis, sum(nelec)))
    phi_uhf = random_complex(rng, (nbasis, sum(nelec)))
    trial_uhf = SingleDet(psi_uhf, nelec, nbasis)
    trial_ghf = SingleDetGHF(trial_uhf)

    walkers_uhf = UHFWalkers(phi_uhf, nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers_uhf.build(trial_uhf)

    ghf_phi = numpy.zeros((2 * nbasis, sum(nelec)), dtype=numpy.complex128)
    ghf_phi[:nbasis, : nelec[0]] = phi_uhf[:, : nelec[0]]
    ghf_phi[nbasis:, nelec[0] :] = phi_uhf[:, nelec[0] :]
    walkers_ghf = GHFWalkers(ghf_phi, nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers_ghf.build(trial_ghf)

    prop_uhf = HubbardSingleSite(timestep)
    prop_uhf.build(ham, trial_uhf, walkers_uhf, walkers_uhf.mpi_handler)
    prop_ghf = HubbardSingleSite(timestep)
    prop_ghf.build(ham, trial_ghf, walkers_ghf, walkers_ghf.mpi_handler)

    for _ in range(nsteps):
        random_fields = rng.random((nbasis, nwalkers))

        prop_uhf.kinetic_importance_sampling(walkers_uhf, trial_uhf)
        prop_ghf.kinetic_importance_sampling(walkers_ghf, trial_ghf)
        prop_uhf._einsum_two_body(walkers_uhf, ham, trial_uhf, random_fields)
        prop_ghf._ghf_two_body(walkers_ghf, ham, trial_ghf, random_fields)
        prop_uhf.kinetic_importance_sampling(walkers_uhf, trial_uhf)
        prop_ghf.kinetic_importance_sampling(walkers_ghf, trial_ghf)

        trial_uhf.calc_greens_function(walkers_uhf, build_full=True)
        trial_ghf.calc_greens_function(walkers_ghf)
        e_uhf = local_energy(system, ham, walkers_uhf, trial_uhf)
        e_ghf = local_energy(system, ham, walkers_ghf, trial_ghf)

        numpy.testing.assert_allclose(walkers_uhf.weight, walkers_ghf.weight, atol=1e-10)
        numpy.testing.assert_allclose(walkers_uhf.ovlp, walkers_ghf.ovlp, atol=1e-10)
        numpy.testing.assert_allclose(e_uhf, e_ghf, atol=1e-10)

        numpy.testing.assert_allclose(
            walkers_ghf.phi[:, :nbasis, nelec[0] :], 0.0, atol=1e-12
        )
        numpy.testing.assert_allclose(
            walkers_ghf.phi[:, nbasis:, : nelec[0]], 0.0, atol=1e-12
        )
        numpy.testing.assert_allclose(
            walkers_uhf.phia, walkers_ghf.phi[:, :nbasis, : nelec[0]], atol=1e-10
        )
        numpy.testing.assert_allclose(
            walkers_uhf.phib, walkers_ghf.phi[:, nbasis:, nelec[0] :], atol=1e-10
        )
        numpy.testing.assert_allclose(walkers_uhf.Ga, walkers_ghf.G[:, :nbasis, :nbasis])
        numpy.testing.assert_allclose(walkers_uhf.Gb, walkers_ghf.G[:, nbasis:, nbasis:])


@pytest.mark.unit
def test_hubbard_local_energy_ghf_matches_block_diagonal_uhf():
    rng = numpy.random.default_rng(11)
    nbasis = 5
    nelec = (2, 2)
    nwalkers = 3
    h = random_complex(rng, (nbasis, nbasis))
    h = h + h.conj().T
    ham = Hubbard(numpy.array([h, h]), U=2.0)
    system = Generic(nelec=nelec)

    psi_uhf = random_complex(rng, (nbasis, sum(nelec)))
    trial_uhf = SingleDet(psi_uhf, nelec, nbasis)
    walkers_uhf = UHFWalkers(psi_uhf, nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers_uhf.build(trial_uhf)
    trial_uhf.calc_greens_function(walkers_uhf, build_full=True)
    e_uhf = local_energy(system, ham, walkers_uhf, trial_uhf)

    psi_ghf = numpy.zeros((2 * nbasis, sum(nelec)), dtype=numpy.complex128)
    psi_ghf[:nbasis, : nelec[0]] = trial_uhf.psi0a
    psi_ghf[nbasis:, nelec[0] :] = trial_uhf.psi0b
    trial_ghf = SingleDetGHF(psi_ghf, nelec, nbasis)
    trial_ghf.calculate_energy(system, ham)
    walkers_ghf = GHFWalkers(psi_ghf, nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers_ghf.build(trial_ghf)
    trial_ghf.calc_greens_function(walkers_ghf)
    e_ghf = local_energy(system, ham, walkers_ghf, trial_ghf)

    numpy.testing.assert_allclose(e_uhf, e_ghf, atol=1e-10)
    numpy.testing.assert_allclose([trial_ghf.energy, trial_ghf.e1b, trial_ghf.e2b], e_ghf[0])
