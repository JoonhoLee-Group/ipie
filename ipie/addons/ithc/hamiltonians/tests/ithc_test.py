# Author: Maxine Luo <man.luo@mpq.mpg.de>,
#         Victor Chen <victor.chen@tum.de>
#
import pytest
import numpy
from ipie.utils.linalg import modified_cholesky
from ipie.addons.ithc.utils.testing import generate_hamiltonian
from ipie.addons.ithc.hamiltonians.generic_ithc import eri_diag
from ipie.addons.ithc.hamiltonians.generic_ithc import mat_decomp
from ipie.addons.ithc.hamiltonians.generic_ithc import GenericITHC


@pytest.mark.unit
def test_eri_diag():
    nmo = 10
    nmo_new = 13
    nelec = (4, 3)
    h1e, chol, nuc, eri = generate_hamiltonian(nmo, nelec, cplx=True, sym=4)
    isometry = 0.005 * (numpy.random.randn(nmo, nmo_new) + 1.0j * numpy.random.randn(nmo, nmo_new))
    W = eri_diag(isometry, eri, tol=1e-13)
    W_eff = mat_decomp(W, tol=None)
    hamil = GenericITHC(h1e, isometry, W)
    return


if __name__ == "__main__":
    test_eri_diag()
