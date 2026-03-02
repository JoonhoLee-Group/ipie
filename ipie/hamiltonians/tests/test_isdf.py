import numpy
import pytest

from ipie.utils.testing import _build_equivalent_molecular_chol_and_isdf


@pytest.mark.unit
def test_molecular_isdf_h1e_mod_matches_reconstructed_chol():
    nmo = 7
    nchol = 19
    nisdf = 23
    ham_chol, ham_isdf, _, _, _ = _build_equivalent_molecular_chol_and_isdf(
        nmo=nmo, nchol=nchol, nisdf=nisdf, seed=13
    )

    numpy.testing.assert_allclose(ham_chol.h1e_mod, ham_isdf.h1e_mod, atol=1e-10)
