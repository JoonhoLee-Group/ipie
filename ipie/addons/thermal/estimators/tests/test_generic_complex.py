import numpy
import pytest
from typing import Tuple, Union

from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.addons.thermal.utils.testing import build_generic_test_case_handlers
from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G

# System params.
nup = 5
ndown = 5
nelec = (nup, ndown)
nbasis = 10

# Thermal AFQMC params.
mu = -10.
beta = 0.1
timestep = 0.01
nwalkers = 12
# Must be fixed at 1 for Thermal AFQMC--legacy code overides whatever input!
nsteps_per_block = 1
nblocks = 12
stabilize_freq = 10
pop_control_freq = 1
pop_control_method = 'pair_branch'
#pop_control_method = 'comb'
lowrank = False

verbose = True
complex_integrals = False
debug = True
mf_trial = True
propagate = False
seed = 7
numpy.random.seed(seed)

options = {
            'nelec': nelec,
            'nbasis': nbasis,
            'mu': mu,
            'beta': beta,
            'timestep': timestep,
            'nwalkers': nwalkers,
            'seed': seed,
            'nsteps_per_block': nsteps_per_block,
            'nblocks': nblocks,
            'stabilize_freq': stabilize_freq,
            'pop_control_freq': pop_control_freq,
            'pop_control_method': pop_control_method,
            'lowrank': lowrank,
            'complex_integrals': complex_integrals,
            'mf_trial': mf_trial,
            'propagate': propagate,
        }


@pytest.mark.unit
def test_local_energy_vs_real():
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    objs =  build_generic_test_case_handlers(options, seed, debug, verbose)
    trial = objs['trial']
    walkers = objs['walkers']
    hamiltonian = objs['hamiltonian']

    assert isinstance(hamiltonian, GenericRealChol)
    
    chol = hamiltonian.chol
    cx_chol = numpy.array(chol, dtype=numpy.complex128)
    cx_hamiltonian = HamGeneric(
        numpy.array(hamiltonian.H1, dtype=numpy.complex128), cx_chol, 
                    hamiltonian.ecore, verbose=False)

    assert isinstance(cx_hamiltonian, GenericComplexChol)
    
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
        energy = local_energy_generic_cholesky(hamiltonian, P)
        cx_energy = local_energy_generic_cholesky(cx_hamiltonian, P)
        numpy.testing.assert_allclose(energy, cx_energy, atol=1e-10)


@pytest.mark.unit
def test_local_energy_vs_eri():
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    options['complex_integrals'] = True
    objs =  build_generic_test_case_handlers(options, seed, debug, with_eri=True,
                                             verbose=verbose)
    trial = objs['trial']
    walkers = objs['walkers']
    hamiltonian = objs['hamiltonian']
    assert isinstance(hamiltonian, GenericComplexChol)
    eri = objs['eri'].reshape(nbasis, nbasis, nbasis, nbasis)
    
    chol = hamiltonian.chol.copy()
    nchol = chol.shape[1]
    chol = chol.reshape(nbasis, nbasis, nchol)

    # Check if chol and eri are consistent.
    eri_chol = numpy.einsum('mnx,slx->mnls', chol, chol.conj())
    numpy.testing.assert_allclose(eri, eri_chol, atol=1e-10)

    for iw in range(walkers.nwalkers):
        print(iw)
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
        Pa, Pb = P
        Ptot = Pa + Pb
        etot, e1, e2 = local_energy_generic_cholesky(hamiltonian, P)

        # Test 1-body term.
        h1e = hamiltonian.H1[0]
        e1ref = numpy.einsum('ij,ij->', h1e, Ptot)
        numpy.testing.assert_allclose(e1, e1ref, atol=1e-10)

        # Test 2-body term.
        ecoul = 0.5 * numpy.einsum('ijkl,ij,kl->', eri, Ptot, Ptot)
        exx = -0.5 * numpy.einsum('ijkl,il,kj->', eri, Pa, Pa)
        exx -= 0.5 * numpy.einsum('ijkl,il,kj->', eri, Pb, Pb)
        e2ref = ecoul + exx
        numpy.testing.assert_allclose(e2, e2ref, atol=1e-10)
        
        etotref = e1ref + e2ref
        numpy.testing.assert_allclose(etot, etotref, atol=1e-10)


if __name__ == '__main__':
    test_local_energy_vs_real()
    test_local_energy_vs_eri()
