import numpy
import pytest

from pyscf import gto, scf, ao2mo
from ipie.qmc.options import QMCOpts
from ipie.utils.linalg import diagonalise_sorted
from ueg import UEG

from ipie.systems.generic import Generic
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.hamiltonians.utils import get_hamiltonian
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.walkers.walkers_dispatch import UHFWalkersTrial
from ipie.estimators.local_energy import local_energy_G

from ipie.legacy.systems.ueg import UEG as LegacyUEG
from ipie.legacy.hamiltonians.ueg import UEG as LegacyHamUEG
from ipie.legacy.trial_wavefunction.free_electron import FreeElectron
from ipie.legacy.trial_wavefunction.multi_slater import MultiSlater
from ipie.legacy.walkers.handler import Walkers
from ipie.legacy.estimators.ueg import local_energy_ueg as legacy_local_energy_ueg
from ipie.legacy.estimators.thermal import one_rdm_from_G as legacy_one_rdm_from_G


def setup_objs(seed=None):
    mu = -10.
    nwalkers = 1
    nblocks = 10
    nsteps_per_block = 1
    timestep = 0.005
    verbose = True
    numpy.random.seed(seed)
    
    options = {
        "qmc": {
            "nwalkers": nwalkers,
            "blocks": nblocks,
            "nsteps": nsteps_per_block,
            "timestep": timestep,
            "rng_seed": seed,
            "batched": False
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
            "thermal": False,
            "write_integrals": False,
            }

    ueg = UEG(ueg_opts, verbose=verbose)
    ueg.build()
    nbasis = ueg.nbasis
    nchol = ueg.nchol
    nelec = (ueg.nup, ueg.ndown)
    nup, ndown = nelec

    h1 = ueg.H1[0]
    chol = 2. * ueg.chol_vecs.toarray()
    #ecore = ueg.ecore
    ecore = 0.

    # -------------------------------------------------------------------------
    # Build trial wavefunction.
    # For pyscf.
    U = ueg.compute_real_transformation()
    h1_8 = U.T.conj() @ h1 @ U
    eri_8 = ueg.eri_8() # 8-fold eri
    eri_8 = ao2mo.restore(8, eri_8, nbasis)
    
    mol = gto.M()
    mol.nelectron = numpy.sum(nelec)
    mol.spin = nup - ndown
    mol.max_memory = 60000 # MB
    mol.incore_anyway = True

    # PW guess.
    dm0a = numpy.zeros(nbasis)
    dm0b = numpy.zeros(nbasis)
    dm0a[:nup] = 1
    dm0b[:ndown] = 1
    dm0 = numpy.array([numpy.diag(dm0a), numpy.diag(dm0b)])

    mf = scf.UHF(mol)
    #mf.level_shift = 0.5
    mf.max_cycle = 5000
    mf.get_hcore = lambda *args: h1_8
    mf.get_ovlp = lambda *args: numpy.eye(nbasis)
    mf._eri = eri_8
    e = mf.kernel(dm0)

    Ca, Cb = mf.mo_coeff
    psia = Ca[:, :nup]
    psib = Cb[:, :ndown]
    psi0 = numpy.zeros((nbasis, numpy.sum(nelec)), dtype=numpy.complex128)
    psi0[:, :nup] = psia
    psi0[:, nup:] = psib
    #numpy.save('ueg_trial', psi0)

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

    # 1. Build out system.
    system = Generic(nelec=nelec, verbose=verbose)
    
    # 2. Build Hamiltonian.
    hamiltonian = HamGeneric(
            numpy.array([h1, h1], dtype=numpy.complex128), 
            numpy.array(chol, dtype=numpy.complex128), 
            ecore,
            verbose=verbose)

    # 3. Build trial.
    trial = SingleDet(psi0, nelec, nbasis, cplx=True, verbose=verbose)
    trial.build()
    trial.half_rotate(hamiltonian)
    trial.calculate_energy(system, hamiltonian)
    if verbose: print(f"\n# trial.energy = {trial.energy}\n")

    # Check against RHF solutions of 10.1063/1.5109572
    assert numpy.allclose(numpy.around(trial.energy, 6), 13.603557) # rs = 1, nbasis = 57
    assert numpy.allclose(trial.energy, e)

    # 4. Build walkers.
    walkers = UHFWalkersTrial(trial, psi0, system.nup, system.ndown, hamiltonian.nbasis, 
                              nwalkers, verbose=verbose)
    walkers.build(trial)
    walkers.ovlp = trial.calc_greens_function(walkers, build_full=True)

    # -------------------------------------------------------------------------
    # Legacy.
    legacy_opts = {
                "trial": {
                    "read_in": "ueg_trial.npy"
                },

                "qmc": options["qmc"],
                "hamiltonian": options["hamiltonian"],
            }

    print('\n------------------------------')
    print('Constructing legacy objects...')
    print('------------------------------')
    
    # 1. Build out system.
    legacy_system = LegacyUEG(options=ueg_opts)

    # 2. Build Hamiltonian.
    legacy_hamiltonian = LegacyHamUEG(legacy_system, options=ueg_opts)
    
    # 3. Build trial.
    coeffs = numpy.array([1])
    psi = numpy.array([psi0])
    wfn = [coeffs, psi]
    legacy_trial = MultiSlater(legacy_system, legacy_hamiltonian, wfn, 
                               nbasis=nbasis, cplx=True, verbose=verbose)

    # 4. Build walkers.
    qmc_opts = QMCOpts()
    qmc_opts.nwalkers = nwalkers
    qmc_opts.ntot_walkers = nwalkers
    qmc_opts.nsteps = nsteps_per_block
    qmc_opts.dt = timestep
    qmc_opts.seed = seed

    legacy_walkers = Walkers(legacy_system, legacy_hamiltonian, legacy_trial,
                             qmc_opts, verbose=verbose)
    
    objs = {'system': system,
            'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers}

    legacy_objs = {'system': legacy_system,
                   'trial': legacy_trial,
                   'hamiltonian': legacy_hamiltonian,
                   'walkers': legacy_walkers}

    return objs, legacy_objs


@pytest.mark.unit
def test_ueg_0T(verbose=False):
    seed = 7
    objs, legacy_objs = setup_objs(seed=seed)
    system = objs['system']
    trial = objs['trial']
    hamiltonian = objs['hamiltonian']
    walkers = objs['walkers']
    
    legacy_system = legacy_objs['system']
    legacy_trial = legacy_objs['trial']
    legacy_hamiltonian = legacy_objs['hamiltonian']
    legacy_walkers = legacy_objs['walkers']
    
    h1e = legacy_hamiltonian.H1[0]
    eri = legacy_hamiltonian.eri_4()
    
    for iw in range(walkers.nwalkers):
        G = numpy.array([walkers.Ga[iw], walkers.Gb[iw]])
        Ghalf = numpy.array([walkers.Ghalfa[iw], walkers.Ghalfb[iw]])
        eloc = local_energy_G(system, hamiltonian, trial, G, Ghalf)
        
        legacy_eloc = legacy_local_energy_ueg(legacy_system, legacy_hamiltonian, legacy_walkers.walkers[iw].G)
        legacy_Ga, legacy_Gb = legacy_walkers.walkers[iw].G
        legacy_Gtot = legacy_Ga + legacy_Gb
        ref_e1 = numpy.einsum('ij,ij->', h1e, legacy_Gtot)

        Gtot = legacy_Gtot
        Ga = legacy_Ga
        Gb = legacy_Gb

        ecoul = 0.5 * numpy.einsum('ijkl,ij,kl->', eri, Gtot, Gtot)
        exx = -0.5 * numpy.einsum('ijkl,il,kj->', eri, Ga, Ga)
        exx -= 0.5 * numpy.einsum('ijkl,il,kj->', eri, Gb, Gb)
        ref_e2 = ecoul + exx
        ref_eloc = (ref_e1 + ref_e2, ref_e1, ref_e2)

        if verbose:
            print(f'\niw = {iw}')
            print(f'eloc = \n{eloc}\n')
            print(f'legacy_eloc = \n{legacy_eloc}\n')
            print(f'ref_eloc = \n{ref_eloc}\n')
            print(f'walkers.weight = \n{walkers.weight[iw]}\n')
            print(f'legacy_walkers.weight = \n{legacy_walkers.walkers[iw].weight}\n')

        numpy.testing.assert_allclose(eloc, ref_eloc, atol=1e-10)
        numpy.testing.assert_allclose(legacy_eloc, ref_eloc, atol=1e-10)
        numpy.testing.assert_almost_equal(legacy_eloc, eloc, decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[0], walkers.Ga[iw], decimal=10)
        numpy.testing.assert_almost_equal(legacy_walkers.walkers[iw].G[1], walkers.Gb[iw], decimal=10)


if __name__ == "__main__":
    test_ueg_0T(verbose=True)
