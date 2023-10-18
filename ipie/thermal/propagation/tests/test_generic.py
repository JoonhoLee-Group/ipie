import numpy
import pytest
import h5py

from pyscf import gto, scf, lo
from ipie.qmc.options import QMCOpts

from ipie.systems.generic import Generic
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.thermal.estimators.thermal import one_rdm_from_G

from ipie.legacy.hamiltonians._generic import Generic as LegacyHamGeneric
from ipie.legacy.trial_density_matrices.mean_field import MeanField as LegacyMeanField
from ipie.legacy.walkers.thermal import ThermalWalker
from ipie.legacy.thermal_propagation.continuous import Continuous
from ipie.legacy.estimators.generic import local_energy_generic_cholesky as legacy_local_energy_generic_cholesky
from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G


def legacy_propagate_walkers(legacy_hamiltonian, legacy_trial, legacy_walkers, legacy_propagator):
    for walker in legacy_walkers:
        legacy_propagator.propagate_walker_phaseless(
                legacy_hamiltonian, walker, legacy_trial)

    return legacy_walkers


def test_phaseless_generic_propagator():
    nocca = 5
    noccb = 5
    nelec = nocca + noccb
    r0 = 1.75
    mol = gto.M(
            atom=[("H", i * r0, 0, 0) for i in range(nelec)],
            basis='sto-6g',
            unit='Bohr',
            verbose=5)

    mf = scf.UHF(mol).run()
    mf.chkfile = 'scf.chk'
    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)
    mf = mf.run(dm1)
    mf.stability(return_status=True)
    s1e = mol.intor("int1e_ovlp_sph")
    ao_coeff = lo.orth.lowdin(s1e)

    path = "/Users/shufay/Documents/in_prep/ft_moire/ipie/ipie/thermal/tests/"
    with h5py.File(path + "reference_data/generic_integrals.h5", "r") as fa:
        Lxmn = fa["LXmn"][:]
        nchol = Lxmn.shape[0]
        nbasis = Lxmn.shape[1]

    mu = -10.
    beta = 0.1
    dt = 0.01
    nwalkers = 1
    seed = 7
    numpy.random.seed(seed)
    blocks = 10
    stabilise_freq = 10
    pop_control_freq = 1
    nsteps = 1
    nslice = 5

    lowrank = False
    verbose = True

    options = {
        "qmc": {
            "dt": dt,
            "nwalkers": nwalkers,
            "blocks": blocks,
            "nsteps": nsteps,
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
            "name": "Generic",
            "integrals": path + "reference_data/generic_integrals.h5",
            "_alt_convention": False,
            "symmetry": False,
            "sparse": False,
            "mu": mu
        },
    }

    # Test.
    print('\n----------------------------')
    print('Constructing test objects...')
    print('----------------------------')
    system = Generic(mol.nelec, verbose=verbose)
    hamiltonian = get_hamiltonian(system, options["hamiltonian"])
    trial = MeanField(hamiltonian, mol.nelec, beta, dt, verbose=verbose)
    nbasis = trial.dmat.shape[-1]
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank, 
                                verbose=verbose)
    propagator = PhaselessGeneric(dt, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)
    
    # Legacy.
    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    legacy_system = Generic(mol.nelec, verbose=verbose)
    legacy_system.mu = mu

    legacy_hamiltonian = LegacyHamGeneric(
                            h1e=hamiltonian.H1,
                            chol=hamiltonian.chol,
                            ecore=hamiltonian.ecore,
                            options=options["hamiltonian"])
    legacy_hamiltonian.hs_pot = numpy.copy(hamiltonian.chol)
    legacy_hamiltonian.hs_pot = legacy_hamiltonian.hs_pot.T.reshape(
            (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
    legacy_hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    legacy_hamiltonian.mu = options["hamiltonian"]["mu"]

    legacy_trial = LegacyMeanField(legacy_system, legacy_hamiltonian, beta, dt, 
                                   verbose=verbose)
    legacy_walkers = [
            ThermalWalker(
                legacy_system, legacy_hamiltonian, legacy_trial, 
                walker_opts=options, verbose=i == 0) for i in range(nwalkers)]

    qmc_opts = QMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers
    qmc_opts.beta = beta
    qmc_opts.nsteps = nsteps
    
    legacy_hamiltonian.chol_vecs = legacy_hamiltonian.chol_vecs.T.reshape(
                    (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
    legacy_propagator = Continuous(
                            options["propagator"], qmc_opts, legacy_system, 
                            legacy_hamiltonian,legacy_trial, verbose=verbose, 
                            lowrank=lowrank)
    
    for t in range(nslice):
        propagator.propagate_walkers(walkers, hamiltonian, trial)
        legacy_walkers = legacy_propagate_walkers(
                            legacy_hamiltonian, legacy_trial, legacy_walkers, legacy_propagator)
        legacy_hamiltonian.hs_pot = legacy_hamiltonian.hs_pot.T.reshape(
                (hamiltonian.nchol, hamiltonian.nbasis, hamiltonian.nbasis))
        
        # Check.
        for iw in range(nwalkers):
            P = one_rdm_from_G(numpy.array([walkers.Ga[iw], walkers.Gb[iw]])) 
            eloc = local_energy_generic_cholesky(hamiltonian, P)

            legacy_P = legacy_one_rdm_from_G(numpy.array(legacy_walkers[iw].G))
            legacy_hamiltonian.chol_vecs = legacy_hamiltonian.chol_vecs.reshape(
                            (hamiltonian.nchol, hamiltonian.nbasis**2)).T
            legacy_eloc = legacy_local_energy_generic_cholesky(
                            legacy_system, legacy_hamiltonian, legacy_P)

            print(f'\nt = {t}')
            print(f'iw = {iw}')
            print(f'eloc = \n{eloc}\n')
            print(f'legacy_eloc = \n{legacy_eloc}\n')

            #numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)
            #numpy.testing.assert_almost_equal(legacy_walkers[iw].G[0], walkers.Ga[iw], decimal=10)
            #numpy.testing.assert_almost_equal(legacy_walkers[iw].G[1], walkers.Gb[iw], decimal=10)
            #numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[0], walkers.stack[iw].ovlp[0], decimal=10)
            #numpy.testing.assert_almost_equal(legacy_walkers[iw].stack.ovlp[1], walkers.stack[iw].ovlp[1], decimal=10)



if __name__ == "__main__":
    test_phaseless_generic_propagator()
