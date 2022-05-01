import numpy
import pytest
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.estimators.generic import (
        local_energy_generic_opt,
        local_energy_cholesky_opt,
        )
from ipie.legacy.estimators.generic import (
        local_energy_generic_cholesky,
        local_energy_generic_cholesky_opt_batched
        )
from ipie.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd
        )

from ipie.utils.pack import pack_cholesky

try:
    import cupy
    no_gpu = not cupy.is_available()
except:
    no_gpu = True


@pytest.mark.unit
@pytest.mark.skipif(no_gpu, reason="gpu not found.")
def test_local_energy_cholesky_opt():
    numpy.random.seed(7)
    nmo = 24
    nelec = (4,2)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)

    chol = chol.reshape((-1,nmo*nmo)).T.copy()

    nchol = chol.shape[-1]
    chol = chol.reshape((nmo,nmo,nchol))

    idx = numpy.triu_indices(nmo)
    cp_shape = (nmo*(nmo+1)//2, chol.shape[-1])
    chol_packed = numpy.zeros(cp_shape, dtype = chol.dtype)
    pack_cholesky(idx[0],idx[1], chol_packed, chol)
    chol = chol.reshape((nmo*nmo,nchol))

    system = Generic(nelec=nelec) 
    ham = HamGeneric (h1e=numpy.array([h1e, h1e]),
                  chol=chol, chol_packed = chol_packed,
                  ecore=enuc)
    wfn = get_random_nomsd(system.nup, system.ndown, ham.nbasis, ndet=1, cplx=False)
    trial = MultiSlater(system, ham, wfn)
    trial.half_rotate(system, ham)
    
    if not no_gpu:
        ham.cast_to_cupy()
        trial.cast_to_cupy()

    e = local_energy_cholesky_opt(system, ham.ecore, trial.Ghalf[0],trial.Ghalf[1], trial)
    
    if not no_gpu:
        e = cupy.array(e)
        e = cupy.asnumpy(e)
    assert e[0] == pytest.approx(20.6826247016273)
    assert e[1] == pytest.approx(23.0173528796140)
    assert e[2] == pytest.approx(-2.3347281779866)

if __name__ == '__main__':
    test_local_energy_cholesky_opt()
