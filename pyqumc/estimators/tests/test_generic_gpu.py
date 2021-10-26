import numpy
import pytest
from pyqumc.systems.generic import Generic
from pyqumc.hamiltonians.generic import Generic as HamGeneric
from pyqumc.trial_wavefunction.multi_slater import MultiSlater
from pyqumc.estimators.generic import (
        local_energy_generic_opt,
        local_energy_generic_cholesky,
        local_energy_generic_cholesky_opt,
        local_energy_generic_cholesky_opt_batched
        )
from pyqumc.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd
        )
try:
    import cupy
    gpu_available = True
except:
    gpu_available = False


@pytest.mark.unit
def test_local_energy_cholesky_opt():
    numpy.random.seed(7)
    nmo = 24
    nelec = (4,2)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec) 
    ham = HamGeneric (h1e=numpy.array([h1e, h1e]),
                  chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                  ecore=enuc)
    wfn = get_random_nomsd(system.nup, system.ndown, ham.nbasis, ndet=1, cplx=False)
    trial = MultiSlater(system, ham, wfn)
    trial.half_rotate(system, ham)
    
    if gpu_available:
        print("# GPU data transfer in trial")
        ham.H1 = cupy.array(ham.H1)
        ham.h1e_mod = cupy.array(ham.h1e_mod)
        ham.chol_vecs = cupy.array(ham.chol_vecs)

        trial.psi = cupy.array(trial.psi)
        trial.coeff = cupy.array(trial.coeff)
        trial._rchola = cupy.array(trial._rchola.copy())
        trial._rcholb = cupy.array(trial._rcholb.copy())
        if (trial.G != None):
            trial.G = cupy.array(trial.G)
        if (trial.Ghalf != None):
            trial.Ghalf[0] = cupy.array(trial.Ghalf[0])
            trial.Ghalf[1] = cupy.array(trial.Ghalf[1])
        if (trial.ortho_expansion):
            trial.occa = cupy.array(trial.occa)
            trial.occb = cupy.array(trial.occb)

    e = local_energy_generic_cholesky_opt(system, ham, trial.G[0], trial.G[1], trial.Ghalf[0],trial.Ghalf[1], trial._rchola, trial._rcholb)
    assert e[0] == pytest.approx(20.6826247016273)
    assert e[1] == pytest.approx(23.0173528796140)
    assert e[2] == pytest.approx(-2.3347281779866)

if __name__ == '__main__':
    test_local_energy_cholesky_opt()