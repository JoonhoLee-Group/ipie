import numpy
import os
import pytest
from pyqumc.systems.generic import Generic
from pyqumc.hamiltonians.generic import Generic as HamGeneric
from pyqumc.trial_wavefunction.multi_slater import MultiSlater
from pyqumc.propagation.generic import GenericContinuous
from pyqumc.propagation.continuous import Continuous
from pyqumc.utils.misc import dotdict
from pyqumc.utils.testing import (
        generate_hamiltonian,
        get_random_nomsd,
        get_random_phmsd
        )
from pyqumc.walkers.multi_det import MultiDetWalker
from pyqumc.estimators.greens_function import gab_spin, gab

@pytest.mark.unit
def test_phmsd_force_bias():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    nchols = chol.shape[0]
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=2, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = GenericContinuous(system, ham, trial, qmc)
    walker = MultiDetWalker(system, ham, trial)
    fb = prop.construct_force_bias(ham, walker, trial)

    ndets = wfn[0].shape[0]
    
    fb_ref = numpy.zeros(nchols, dtype=numpy.complex128)

    nbsf = ham.nbasis
    G = numpy.zeros((nbsf,nbsf), dtype=numpy.complex128)
    denom = 0.0 + 0.0j
    for idet in range(ndets):
        Gi, Gihalf = gab_spin(trial.psi[idet], walker.phi, nelec[0], nelec[1])
        Gia = Gi[0]
        Gib = Gi[1]
        Oia = numpy.dot(trial.psi[idet][:,:nelec[0]].conj().T, walker.phi[:,:nelec[0]])
        Oib = numpy.dot(trial.psi[idet][:,nelec[0]:].conj().T, walker.phi[:,nelec[0]:])
        
        sign_a, logdet_a = numpy.linalg.slogdet(Oia)
        sign_b, logdet_b = numpy.linalg.slogdet(Oib)
        ovlpa = sign_a*numpy.exp(logdet_a)  
        ovlpb = sign_b*numpy.exp(logdet_b)
        ovlp = ovlpa * ovlpb
        G += Gia * ovlpa * trial.coeffs[idet].conj() + Gib * ovlpb * trial.coeffs[idet].conj()
        denom += ovlp * trial.coeffs[idet].conj()

    G /= denom
    fb_ref[:] = - prop.sqrt_dt*(1.j*numpy.dot(ham.chol_vecs.T, G.ravel())- prop.mf_shift)
    
    assert numpy.allclose(fb_ref, fb)

@pytest.mark.unit
def test_phmsd_overlap():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    nchols = chol.shape[0]
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=2, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = GenericContinuous(system, ham, trial, qmc)
    walker = MultiDetWalker(system, ham, trial)

    ndets = wfn[0].shape[0]

    nbsf = ham.nbasis
    denom = 0.0 + 0.0j
    for idet in range(ndets):
        Gi, Gihalf = gab_spin(trial.psi[idet], walker.phi, nelec[0], nelec[1])
        Gia = Gi[0]
        Gib = Gi[1]
        Oia = numpy.dot(trial.psi[idet][:,:nelec[0]].conj().T, walker.phi[:,:nelec[0]])
        Oib = numpy.dot(trial.psi[idet][:,nelec[0]:].conj().T, walker.phi[:,nelec[0]:])
        
        sign_a, logdet_a = numpy.linalg.slogdet(Oia)
        sign_b, logdet_b = numpy.linalg.slogdet(Oib)
        ovlpa = sign_a*numpy.exp(logdet_a)  
        ovlpb = sign_b*numpy.exp(logdet_b)
        ovlp = ovlpa * ovlpb
        denom += ovlp * trial.coeffs[idet].conj()
    
    assert numpy.allclose(denom, walker.ovlp)

@pytest.mark.unit
def test_phmsd_propagation():
    numpy.random.seed(7)
    nmo = 10
    nelec = (5,5)
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=0)
    # Test PH type wavefunction.
    wfn, init = get_random_phmsd(system.nup, system.ndown, ham.nbasis, ndet=3, init=True)
    trial = MultiSlater(system, ham, wfn, init=init)
    trial.calculate_energy(system, ham)
    options = {'hybrid': False}
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = Continuous(system, ham, trial, qmc, options=options)
    walker = MultiDetWalker(system, ham, trial)
    for i in range(0,10):
        prop.propagate_walker(walker, system, ham, trial, trial.energy)
    assert walker.weight == pytest.approx(0.6689207316889051)


if __name__ == '__main__':
    test_phmsd_force_bias()
    test_phmsd_overlap()
    test_phmsd_propagation()
