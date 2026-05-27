import numpy

from ipie.utils.testing import *  # noqa: F401,F403
from ipie.utils.testing import generate_hamiltonian, get_random_nomsd, shaped_normal
from ipie.addons.ithc.hamiltonians.generic_ithc import GenericITHC, eri_diag
from ipie.addons.ithc.trial_wavefunction.single_det import SingleDet
from ipie.addons.ithc.walkers.walkers_dispatch import UHFWalkersTrial
from ipie.systems import Generic
from ipie.utils.mpi import MPIHandler


def gen_random_test_instances_ithc(nmo, nocc, naux, nwalkers, seed=7, ndets=1):
    assert ndets == 1
    numpy.random.seed(seed)
    wfn = get_random_nomsd(nocc, nocc, nmo, ndet=ndets)
    h1e, _, _, eri = generate_hamiltonian(nmo, nocc, cplx=True, sym=4)
    isometry = 0.005 * (numpy.random.randn(nmo, naux) + 1.0j * numpy.random.randn(nmo, naux))
    W = eri_diag(isometry, eri, tol=1e-13)

    system = Generic(nelec=(nocc, nocc))
    ham = GenericITHC(h1e, isometry, W)

    trial = SingleDet(wfn[1][0], (nocc, nocc), nmo)
    walkers = UHFWalkersTrial(
        trial,
        wfn[1][0],
        system.nup,
        system.ndown,
        ham.nbasis,
        nwalkers,
        MPIHandler(),
    )
    walkers.build(trial)

    walkers.Ghalfa = shaped_normal((nwalkers, nocc, nmo), cmplx=True)
    walkers.Ghalfb = shaped_normal((nwalkers, nocc, nmo), cmplx=True)
    trial._rchola = shaped_normal((naux, nocc * nmo))
    trial._rcholb = shaped_normal((naux, nocc * nmo))
    trial._rH1a = shaped_normal((nocc, nmo))
    trial._rH1b = shaped_normal((nocc, nmo))
    return system, ham, walkers, trial
