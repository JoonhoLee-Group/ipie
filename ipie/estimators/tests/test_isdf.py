import numpy
import pytest

from ipie.config import config
from ipie.estimators.local_energy_sd import local_energy_single_det_batch_gpu
from ipie.estimators.local_energy_sd_isdf import local_energy_single_det_isdf_batch_gpu
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import _build_equivalent_molecular_chol_and_isdf
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.utils.backend import arraylib as xp


@pytest.mark.gpu
def test_molecular_isdf_local_energy_matches_reconstructed_chol():
    if not config.get_option("use_gpu"):
        pytest.skip("Requires GPU backend. Set IPIE_USE_GPU=1 to run this test.")

    nmo = 8
    nelec = (3, 3)
    nwalkers = 4
    system = Generic(nelec=nelec)
    ham_chol, ham_isdf, _, _, _ = _build_equivalent_molecular_chol_and_isdf(
        nmo=nmo, nchol=20, nisdf=24, seed=31
    )

    psi = numpy.hstack([numpy.eye(nmo)[:, : nelec[0]], numpy.eye(nmo)[:, : nelec[1]]])
    trial_chol = SingleDet(psi, nelec, nmo)
    trial_isdf = SingleDet(psi, nelec, nmo)
    trial_chol.half_rotate(ham_chol)
    trial_isdf.half_rotate(ham_isdf)

    walkers = UHFWalkers(psi, nelec[0], nelec[1], nmo, nwalkers, MPIHandler())
    walkers.build(trial_chol)

    ham_chol.cast_to_cupy()
    ham_isdf.cast_to_cupy()
    trial_chol.cast_to_cupy()
    trial_isdf.cast_to_cupy()
    walkers.cast_to_cupy()

    e_chol = local_energy_single_det_batch_gpu(system, ham_chol, walkers, trial_chol)
    e_isdf = local_energy_single_det_isdf_batch_gpu(system, ham_isdf, walkers, trial_isdf)

    xp.testing.assert_allclose(e_isdf, e_chol, atol=1e-8)
