import numpy
import pytest

from ipie.estimators.greens_function_single_det import greens_function_single_det_ghf
from ipie.trial_wavefunction.single_det_ghf import SingleDetGHF
from ipie.utils.mpi import MPIHandler
from ipie.walkers.ghf_walkers import GHFWalkers


@pytest.mark.unit
@pytest.mark.parametrize("real_trial", [False, True])
def test_ghf_greens_function_cpu_matches_direct(real_trial):
    rng = numpy.random.default_rng(5)
    nbasis, nelec, nwalkers = 6, (3, 2), 4
    nocc = sum(nelec)
    psi0 = rng.normal(size=(2 * nbasis, nocc))
    if not real_trial:
        psi0 = psi0 + 1j * rng.normal(size=psi0.shape)
    trial = SingleDetGHF(psi0, nelec, nbasis)
    walkers = GHFWalkers(psi0.astype(complex), nelec[0], nelec[1], nbasis, nwalkers, MPIHandler())
    walkers.phi = rng.normal(size=walkers.phi.shape) + 1j * rng.normal(size=walkers.phi.shape)
    walkers.build(trial)
    walkers.log_shift = rng.normal(size=nwalkers)

    det = greens_function_single_det_ghf(walkers, trial)

    for iw in range(nwalkers):
        ovlp = psi0.conj().T @ walkers.phi[iw]
        G_ref = (walkers.phi[iw] @ numpy.linalg.solve(ovlp, psi0.conj().T)).T
        det_ref = numpy.linalg.det(ovlp) * numpy.exp(-walkers.log_shift[iw])
        numpy.testing.assert_allclose(walkers.G[iw], G_ref, rtol=1e-10, atol=1e-12)
        numpy.testing.assert_allclose(det[iw], det_ref, rtol=1e-10)
    numpy.testing.assert_allclose(walkers.Ga, walkers.G[:, :nbasis, :nbasis])
    numpy.testing.assert_allclose(walkers.Gb, walkers.G[:, nbasis:, nbasis:])
