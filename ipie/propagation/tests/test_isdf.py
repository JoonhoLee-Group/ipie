import numpy
import pytest

from ipie.propagation.phaseless_base import (
    construct_mean_field_shift,
    construct_one_body_propagator,
)
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.testing import _build_equivalent_molecular_chol_and_isdf


@pytest.mark.unit
def test_molecular_isdf_mean_field_shift_matches_reconstructed_chol():
    nmo = 8
    nelec = (3, 3)
    ham_chol, ham_isdf, _, _, _ = _build_equivalent_molecular_chol_and_isdf(
        nmo=nmo, nchol=20, nisdf=24, seed=37
    )

    psi = numpy.hstack([numpy.eye(nmo)[:, : nelec[0]], numpy.eye(nmo)[:, : nelec[1]]])
    trial = SingleDet(psi, nelec, nmo)

    mf_chol = construct_mean_field_shift(ham_chol, trial)
    mf_isdf = construct_mean_field_shift(ham_isdf, trial)

    numpy.testing.assert_allclose(mf_isdf, mf_chol, atol=1e-10)


@pytest.mark.unit
def test_molecular_isdf_one_body_propagator_matches_reconstructed_chol():
    nmo = 7
    nelec = (3, 2)
    dt = 0.005
    ham_chol, ham_isdf, _, _, _ = _build_equivalent_molecular_chol_and_isdf(
        nmo=nmo, nchol=18, nisdf=22, seed=41
    )

    psi = numpy.hstack([numpy.eye(nmo)[:, : nelec[0]], numpy.eye(nmo)[:, : nelec[1]]])
    trial = SingleDet(psi, nelec, nmo)

    mf_chol = construct_mean_field_shift(ham_chol, trial)
    mf_isdf = construct_mean_field_shift(ham_isdf, trial)
    expH1_chol = construct_one_body_propagator(ham_chol, mf_chol, dt)
    expH1_isdf = construct_one_body_propagator(ham_isdf, mf_isdf, dt)

    numpy.testing.assert_allclose(expH1_isdf, expH1_chol, atol=1e-10)
