import numpy
import pytest
from pie.hamiltonians.hubbard import Hubbard
from pie.systems.generic import Generic
from pie.propagation.hubbard import Hirsch
from pie.trial_wavefunction.multi_slater import MultiSlater
from pie.walkers.single_det import SingleDetWalker
from pie.walkers.single_det_batch import SingleDetWalkerBatch
from pie.utils.misc import dotdict
from pie.propagation.overlap import calc_overlap_single_det
from pie.estimators.greens_function import gab
from pie.estimators.local_energy import local_energy
from pie.estimators.local_energy_batch import local_energy_batch
from pie.estimators.greens_function import greens_function_single_det

@pytest.mark.unit
def test_buff_size_batch():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Generic((8,8), verbose=False)
    ham = Hubbard(options, verbose=False)
    eigs, eigv = numpy.linalg.eigh(ham.H1[0])
    coeffs = numpy.array([1.0+0j])
    wfn = numpy.zeros((1,ham.nbasis,system.ne))
    wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
    wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
    trial = MultiSlater(system, ham, (coeffs, wfn))
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]

    nwalkers = 1
    walkers = [SingleDetWalker(system, ham, trial) for i in range(nwalkers)]

    nwalkers = 1
    walkers_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)
    
@pytest.mark.unit
def test_overlap_batch():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Generic((8,8), verbose=False)
    ham = Hubbard(options, verbose=False)
    eigs, eigv = numpy.linalg.eigh(ham.H1[0])
    coeffs = numpy.array([1.0+0j])
    wfn = numpy.zeros((1,ham.nbasis,system.ne))
    wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
    wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
    trial = MultiSlater(system, ham, (coeffs, wfn))
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]

    nwalkers = 10
    walkers = [SingleDetWalker(system, ham, trial) for i in range(nwalkers)]

    nup = system.nup
    for iw, walker in enumerate(walkers):
        ovlp = numpy.dot(trial.psi[:,:nup].conj().T, walkers[iw].phi[:,:nup])
        id_exp = numpy.dot(walkers[iw].inv_ovlp[0], ovlp)
        numpy.testing.assert_allclose(id_exp, numpy.eye(nup), atol=1e-12)

    nwalkers = 10
    walkers_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)
    for iw in range(nwalkers):
        # ovlp = numpy.dot(trial.psi[:,:nup].conj().T, walkers_batch.phi[iw][:,:nup])
        # id_exp = numpy.dot(walkers_batch.inv_ovlpa[iw], ovlp)
        # numpy.testing.assert_allclose(id_exp, numpy.eye(nup), atol=1e-12)
        numpy.testing.assert_allclose(walkers[iw].Ghalf[0], walkers_batch.Ghalfa[iw], atol=1e-12)
        numpy.testing.assert_allclose(walkers[iw].Ghalf[1], walkers_batch.Ghalfb[iw], atol=1e-12)
        numpy.testing.assert_allclose(walkers[iw].G[0], walkers_batch.Ga[iw], atol=1e-12)
        numpy.testing.assert_allclose(walkers[iw].G[1], walkers_batch.Gb[iw], atol=1e-12)


@pytest.mark.unit
def test_greens_function_batch():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Generic((8,8), verbose=False)
    ham = Hubbard(options, verbose=False)
    eigs, eigv = numpy.linalg.eigh(ham.H1[0])
    coeffs = numpy.array([1.0+0j])
    wfn = numpy.zeros((1,ham.nbasis,system.ne))
    wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
    wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
    trial = MultiSlater(system, ham, (coeffs, wfn))
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    nup = system.nup
    walker = SingleDetWalker(system, ham, trial)

    nwalkers = 1
    walkers_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)

    # Test Green's function
    Gref = gab(trial.psi[:,:nup], walker.phi[:,:nup])
    numpy.testing.assert_allclose(walker.G[0], Gref, atol=1e-12)
    numpy.testing.assert_allclose(walker.G[0], walkers_batch.Ga[0], atol=1e-12)

@pytest.mark.unit
def test_reortho_batch():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Generic((8,8), verbose=False)
    ham = Hubbard(options, verbose=False)
    eigs, eigv = numpy.linalg.eigh(ham.H1[0])
    coeffs = numpy.array([1.0+0j])
    numpy.random.seed(7)
    wfn = numpy.random.random((ham.nbasis*system.ne)).reshape(1,ham.nbasis, system.ne)
    trial = MultiSlater(system, ham, (coeffs, wfn))
    trial.psi = trial.psi[0]
    trial.psia = trial.psia[0]
    trial.psib = trial.psib[0]
    nup = system.nup
    walker = SingleDetWalker(system, ham, trial)

    nwalkers = 1
    walkers_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)

    # Test Green's function
    ovlp = walker.calc_overlap(trial)
    ovlp_batch = calc_overlap_single_det(walkers_batch,trial)

    assert walker.ot == pytest.approx(ovlp)
    assert walker.ot == pytest.approx(ovlp_batch[0])
    assert walker.ot == pytest.approx(walkers_batch.ot[0])

    eloc = local_energy(system, ham, walker, trial)
    detR = walker.reortho(trial)
    eloc_new = local_energy(system, ham, walker, trial)

    eloc_batch = local_energy_batch(system, ham, walkers_batch, trial)
    detR_batch = walkers_batch.reortho()
    eloc_new_batch = local_energy_batch(system, ham, walkers_batch, trial)

    assert eloc == pytest.approx(eloc_new)
    assert detR*walker.ot == pytest.approx(ovlp)

    assert eloc_batch[0] == pytest.approx(eloc)
    assert eloc_batch[0] == pytest.approx(eloc_new_batch[0])
    assert detR_batch[0]*walkers_batch.ot[0] == pytest.approx(ovlp_batch[0])
    assert detR_batch[0]*walkers_batch.ot[0] == pytest.approx(ovlp)

if __name__=="__main__":
    test_overlap_batch()
    test_greens_function_batch()
    test_reortho_batch()
    test_buff_size_batch()
