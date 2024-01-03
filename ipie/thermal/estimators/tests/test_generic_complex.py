import numpy
import pytest
from typing import Tuple, Union

from ipie.utils.misc import dotdict
from ipie.utils.testing import generate_hamiltonian
from ipie.utils.testing import build_test_case_handlers as build_test_case_handlers_0T
from ipie.estimators.energy import local_energy

from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.thermal.estimators.thermal import one_rdm_from_G


def build_test_case_handlers(nelec: Tuple[int, int],
                             options: Union[dict, None] = None,
                             seed: Union[int, None] = None,
                             choltol: float = 1e-3,
                             complex_integrals: bool = False,
                             verbose: bool = False):
    if seed is not None:
        numpy.random.seed(seed)

    # Unpack options
    mu = options['mu']
    nbasis = options['nbasis']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    lowrank = options['lowrank']
    
    sym = 8
    if complex_integrals: sym = 4
    h1e, chol, _, eri = generate_hamiltonian(nbasis, nelec, cplx=complex_integrals, 
                                             sym=sym, tol=choltol)

    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]),
                             chol=chol.reshape((-1, nbasis**2)).T.copy(),
                             ecore=0)
    hamiltonian.name = options['hamiltonian']['name']
    hamiltonian._alt_convention = options['hamiltonian']['_alt_convention']
    hamiltonian.sparse = options['hamiltonian']['sparse']
    hamiltonian.eri = eri.copy()

    trial = MeanField(hamiltonian, nelec, beta, timestep)
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank)
    propagator = PhaselessGeneric(timestep, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)
        
    for t in range(walkers.stack[0].nslice):
        propagator.propagate_walkers(walkers, hamiltonian, trial)
    
    objs = {'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers,
            'propagator': propagator}
    return objs


@pytest.mark.unit
def test_local_energy_vs_real():
    nocca = 5
    noccb = 5
    nelec = (nocca, noccb)
    nbasis = 10

    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 1
    seed = 7
    lowrank = False
    verbose = True
    
    options = {
                'mu': mu,
                'nbasis': nbasis,
                'beta': beta,
                'timestep': timestep,
                'nwalkers': nwalkers,
                'seed': 7,
                'lowrank': lowrank,

                'hamiltonian': {
                    'name': 'Generic',
                    '_alt_convention': False,
                    'sparse': False,
                    'mu': mu
                }
            }
    
    objs = build_test_case_handlers(nelec, options, seed, choltol=1e-10, 
                                    complex_integrals=False, verbose=verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']

    chol = hamiltonian.chol
    cx_chol = numpy.array(chol, dtype=numpy.complex128)
    cx_hamiltonian = HamGeneric(
        numpy.array(hamiltonian.H1, dtype=numpy.complex128), cx_chol, 
                    hamiltonian.ecore, verbose=False)
    
    for iw in range(walkers.nwalkers):
        P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
        energy = local_energy_generic_cholesky(hamiltonian, P)
        cx_energy = local_energy_generic_cholesky(cx_hamiltonian, P)
        numpy.testing.assert_allclose(energy, cx_energy, atol=1e-10)


@pytest.mark.unit
def test_local_energy_vs_eri():
    nocca = 5
    noccb = 5
    nelec = (nocca, noccb)
    nbasis = 10

    mu = -10.0
    beta = 0.1
    timestep = 0.01
    nwalkers = 1
    seed = 7
    lowrank = False
    verbose = True
    
    options = {
                'mu': mu,
                'nbasis': nbasis,
                'beta': beta,
                'timestep': timestep,
                'nwalkers': nwalkers,
                'seed': 7,
                'lowrank': lowrank,

                'hamiltonian': {
                    'name': 'Generic',
                    '_alt_convention': False,
                    'sparse': False,
                    'mu': mu
                }
            }
    
    objs = build_test_case_handlers(nelec, options, seed, choltol=1e-10, 
                                    complex_integrals=False, verbose=verbose)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']

    eri = hamiltonian.eri.reshape(nbasis, nbasis, nbasis, nbasis)
    chol = hamiltonian.chol.copy()
    nchol = chol.shape[1]
    chol = chol.reshape(nbasis, nbasis, nchol)

    # Check if chol and eri are consistent.
    eri_chol = numpy.einsum('mnx,slx->mnls', chol, chol.conj())
    numpy.testing.assert_allclose(eri, eri_chol, atol=1e-10)

    for iw in range(walkers.nwalkers):
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


@pytest.mark.unit
def test_local_energy_0T_single_det():
    numpy.random.seed(7)
    nmo = 10
    nelec = (6, 5)
    nwalkers = 1
    nsteps = 25
    qmc = dotdict(
        {
            "dt": 0.005,
            "nstblz": 5,
            "nwalkers": nwalkers,
            "hybrid": True,
            "num_steps": nsteps,
        }
    )

    handler_0T = build_test_case_handlers_0T(
        nelec,
        nmo,
        num_dets=1,
        options=qmc,
        seed=7,
        complex_integrals=True,
        complex_trial=True,
        trial_type="single_det",
        choltol=1e-10,
    )

    hamiltonian = handler_0T.hamiltonian
    walkers = handler_0T.walkers
    trial = handler_0T.trial
    walkers.ovlp = trial.calc_greens_function(walkers, build_full=True)
    energy = local_energy(hamiltonian, walkers, trial)
    test_energy = numpy.array(
                    [local_energy_generic_cholesky(
                        hamiltonian, 
                        numpy.array([walkers.Ga[0], walkers.Gb[0]]))])

    print(f'\n0T energy = \n{energy}\n')
    print(f'test_energy = \n{test_energy}\n')
    
    numpy.testing.assert_allclose(energy, test_energy, atol=1e-10)


if __name__ == '__main__':
    test_local_energy_vs_real()
    test_local_energy_vs_eri()
    test_local_energy_0T_single_det()
