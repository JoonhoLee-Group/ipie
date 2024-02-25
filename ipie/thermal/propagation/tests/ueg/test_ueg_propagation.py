import pytest
import numpy

from ueg import UEG
from pyscf import gto, scf, ao2mo
from ipie.qmc.options import QMCOpts

from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.thermal.estimators.thermal import one_rdm_from_G

from ipie.legacy.systems.ueg import UEG as LegacyUEG
from ipie.legacy.hamiltonians.ueg import UEG as LegacyHamUEG
from ipie.legacy.trial_density_matrices.onebody import OneBody as LegacyOneBody
from ipie.legacy.walkers.thermal import ThermalWalker
from ipie.legacy.thermal_propagation.planewave import PlaneWave
from ipie.legacy.estimators.ueg import local_energy_ueg as legacy_local_energy_ueg
from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G


def legacy_propagate_walkers(legacy_hamiltonian, legacy_trial, legacy_walkers, 
                             legacy_propagator, lowrank=False, xi=None):
    if xi is None:
        xi = [None] * legacy_walkers.nwalker

    for iw, walker in enumerate(legacy_walkers):
        if lowrank:
            legacy_propagator.propagate_walker_phaseless_low_rank(
                    legacy_hamiltonian, walker, legacy_trial, xi=xi[iw])
        
        else:
            legacy_propagator.propagate_walker_phaseless_full_rank(
                    legacy_hamiltonian, walker, legacy_trial, xi=xi[iw])
        

    return legacy_walkers


def setup_objs(seed=None):
    mu = -10.
    beta = 0.01
    timestep = 0.002
    nwalkers = 3
    nblocks = 2
    stabilise_freq = 10
    pop_control_freq = 1
    nsteps_per_block = 1

    lowrank = False
    verbose = True
    numpy.random.seed(seed)
    
    options = {
        "qmc": {
            "dt": timestep,
            "nwalkers": nwalkers,
            "blocks": nblocks,
            "nsteps": nsteps_per_block,
            "beta": beta,
            "rng_seed": seed,
            "pop_control_freq": pop_control_freq,
            "stabilise_freq": stabilise_freq,
            "batched": False
        },

        "propagator": {
            "optimised": False
        },

        "walkers": {
            "low_rank": lowrank
        },

        "hamiltonian": {
            "name": "UEG",
            "_alt_convention": False,
            "sparse": False,
            "mu": mu
        }
    }

    # Generate UEG integrals.
    ueg_opts = {
            "nup": 7,
            "ndown": 7,
            "rs": 1.,
            "ecut": 2.5,
            "thermal": True,
            "write_integrals": False,
            "low_rank": lowrank
            }

    ueg = UEG(ueg_opts, verbose=verbose)
    ueg.build(verbose=verbose)
    nbasis = ueg.nbasis
    nchol = ueg.nchol
    nelec = (ueg.nup, ueg.ndown)
    nup, ndown = nelec

    h1 = ueg.H1[0]
    chol = 2. * ueg.chol_vecs.toarray().copy()
    #ecore = ueg.ecore
    ecore = 0.

    if verbose:
        print(numpy.amax(numpy.absolute(chol.imag)))
        print(f"# nbasis = {nbasis}")
        print(f"# nchol = {nchol}")
        print(f"# nup = {nup}")
        print(f"# ndown = {ndown}")

    # -------------------------------------------------------------------------
    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    
    # 1. Build Hamiltonian.
    hamiltonian = HamGeneric(
            numpy.array([h1, h1], dtype=numpy.complex128), 
            numpy.array(chol, dtype=numpy.complex128), 
            ecore,
            verbose=verbose)
    hamiltonian.name = options["hamiltonian"]["name"]
    hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    hamiltonian.sparse = options["hamiltonian"]["sparse"]

    # 2. Build trial.
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # 3. Build walkers.
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank, 
                                verbose=verbose)

    # 4. Build propagator.
    propagator = PhaselessGeneric(timestep, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)

    # -------------------------------------------------------------------------
    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    
    # 1. Build out system.
    legacy_system = LegacyUEG(options=ueg_opts)
    
    # 2. Build Hamiltonian.
    legacy_hamiltonian = LegacyHamUEG(legacy_system, options=ueg_opts)
    legacy_hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    legacy_hamiltonian.mu = options["hamiltonian"]["mu"]

    # 3. Build trial.
    legacy_trial = LegacyOneBody(legacy_system, legacy_hamiltonian, beta, timestep, verbose=verbose)
    
    # 4. Build walkers.
    legacy_walkers = [
            ThermalWalker(
                legacy_system, legacy_hamiltonian, legacy_trial,
                walker_opts=options, verbose=i == 0) for i in range(nwalkers)]

    # 5. Build propagator.
    qmc_opts = QMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers
    qmc_opts.beta = beta
    qmc_opts.nsteps = nsteps_per_block
    qmc_opts.dt = timestep
    qmc_opts.seed = seed

    legacy_propagator = PlaneWave(legacy_system, legacy_hamiltonian, legacy_trial, qmc_opts,
                                  options=options["propagator"], lowrank=lowrank, verbose=verbose)
    
    objs = {'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers,
            'propagator': propagator}

    legacy_objs = {'system': legacy_system,
                   'trial': legacy_trial,
                   'hamiltonian': legacy_hamiltonian,
                   'walkers': legacy_walkers,
                   'propagator': legacy_propagator}

    return objs, legacy_objs


@pytest.mark.unit
def test_phaseless_generic_propagator(verbose=False):
    seed = 7
    objs, legacy_objs = setup_objs(seed=seed)
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    propagator = objs['propagator']
    
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    legacy_propagator = legacy_objs['propagator']

    h1e = legacy_hamiltonian.H1[0]
    eri = legacy_hamiltonian.eri_4()

    for t in range(walkers.stack[0].nslice):
        for iw in range(walkers.nwalkers):
            P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
            eloc = local_energy_generic_cholesky(hamiltonian, P)

            legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers[iw].G))
            legacy_eloc = legacy_local_energy_ueg(legacy_system, legacy_hamiltonian, legacy_P)
            
            legacy_Pa, legacy_Pb = legacy_P
            legacy_Ptot = legacy_Pa + legacy_Pb
            ref_e1 = numpy.einsum('ij,ij->', h1e, legacy_Ptot)

            Ptot = legacy_Ptot
            Pa = legacy_Pa
            Pb = legacy_Pb

            ecoul = 0.5 * numpy.einsum('ijkl,ij,kl->', eri, Ptot, Ptot)
            exx = -0.5 * numpy.einsum('ijkl,il,kj->', eri, Pa, Pa)
            exx -= 0.5 * numpy.einsum('ijkl,il,kj->', eri, Pb, Pb)
            ref_e2 = ecoul + exx
            ref_eloc = (ref_e1 + ref_e2, ref_e1, ref_e2)
        
            if verbose:
                print(f'\nt = {t}')
                print(f'iw = {iw}')
                print(f'eloc = \n{eloc}\n')
                print(f'legacy_eloc = \n{legacy_eloc}\n')
                print(f'ref_eloc = \n{ref_eloc}\n')
                print(f'walkers.weight = \n{walkers.weight[iw]}\n')
                print(f'legacy_walkers.weight = \n{legacy_walkers[iw].weight}\n')

            numpy.testing.assert_almost_equal(legacy_P, P, decimal=10)
            numpy.testing.assert_almost_equal(legacy_trial.dmat, trial.dmat, decimal=10)
            numpy.testing.assert_allclose(eloc, ref_eloc, atol=1e-10)
            numpy.testing.assert_allclose(legacy_eloc, ref_eloc, atol=1e-10)
            numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)

            numpy.testing.assert_almost_equal(legacy_walkers[iw].G[0], walkers.Ga[iw], decimal=10)
            numpy.testing.assert_almost_equal(legacy_walkers[iw].G[1], walkers.Gb[iw], decimal=10)
            numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
            numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)

        propagator.propagate_walkers(walkers, hamiltonian, trial, debug=True)
        legacy_walkers = legacy_propagate_walkers(
                            legacy_hamiltonian, legacy_trial, legacy_walkers, 
                            legacy_propagator, xi=propagator.xi)


if __name__ == "__main__":
    test_phaseless_generic_propagator(verbose=True)
