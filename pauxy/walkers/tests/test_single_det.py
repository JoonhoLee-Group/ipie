import numpy
import pytest
from pauxy.systems.hubbard import Hubbard
from pauxy.propagation.hubbard import Hirsch
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.walkers.single_det import SingleDetWalker
from pauxy.utils.misc import dotdict
from pauxy.estimators.greens_function import gab


@pytest.mark.unit
def test_overlap():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Hubbard(inputs=options)
    eigs, eigv = numpy.linalg.eigh(system.H1[0])
    coeffs = numpy.array([1.0+0j])
    wfn = numpy.zeros((1,system.nbasis,system.ne))
    wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
    wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
    trial = MultiSlater(system, (coeffs, wfn))
    trial.psi = trial.psi[0]
    walker = SingleDetWalker(system, trial)
    nup = system.nup
    # Test overlap
    ovlp = numpy.dot(trial.psi[:,:nup].conj().T, walker.phi[:,:nup])
    id_exp = numpy.dot(walker.inv_ovlp[0], ovlp)
    numpy.testing.assert_allclose(id_exp, numpy.eye(nup), atol=1e-12)

@pytest.mark.unit
def test_update_overlap():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Hubbard(inputs=options)
    eigs, eigv = numpy.linalg.eigh(system.H1[0])
    coeffs = numpy.array([1.0+0j])
    wfn = numpy.zeros((1,system.nbasis,system.ne))
    wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
    wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
    trial = MultiSlater(system, (coeffs, wfn))
    trial.psi = trial.psi[0]
    walker = SingleDetWalker(system, trial)
    # Test update
    nup = system.nup
    vtup = walker.phi[3,:nup] * 0.333
    walker.phi[3,:nup] = walker.phi[3,:nup] + vtup
    vtdn = walker.phi[3,nup:] * -0.333
    walker.phi[3,nup:] = walker.phi[3,nup:] + vtdn
    walker.update_inverse_overlap(trial, vtup, vtdn, 3)
    ovlp = numpy.dot(trial.psi[:,:nup].conj().T, walker.phi[:,:nup])
    id_exp = numpy.dot(walker.inv_ovlp[0], ovlp)
    numpy.testing.assert_allclose(id_exp, numpy.eye(nup), atol=1e-12)
    ovlp = numpy.dot(trial.psi[:,nup:].conj().T, walker.phi[:,nup:])
    id_exp = numpy.dot(walker.inv_ovlp[1], ovlp)
    numpy.testing.assert_allclose(id_exp, numpy.eye(nup), atol=1e-12)

@pytest.mark.unit
def test_greens_function():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Hubbard(inputs=options)
    eigs, eigv = numpy.linalg.eigh(system.H1[0])
    coeffs = numpy.array([1.0+0j])
    wfn = numpy.zeros((1,system.nbasis,system.ne))
    wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
    wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
    trial = MultiSlater(system, (coeffs, wfn))
    trial.psi = trial.psi[0]
    nup = system.nup
    walker = SingleDetWalker(system, trial)
    # Test Green's function
    Gref = gab(trial.psi[:,:nup], walker.phi[:,:nup])
    numpy.testing.assert_allclose(walker.G[0], Gref, atol=1e-12)

@pytest.mark.unit
def test_reortho():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Hubbard(inputs=options)
    eigs, eigv = numpy.linalg.eigh(system.H1[0])
    coeffs = numpy.array([1.0+0j])
    numpy.random.seed(7)
    wfn = numpy.random.random((system.nbasis*system.ne)).reshape(1,system.nbasis, system.ne)
    trial = MultiSlater(system, (coeffs, wfn))
    trial.psi = trial.psi[0]
    nup = system.nup
    walker = SingleDetWalker(system, trial)
    # Test Green's function
    ovlp = walker.calc_overlap(trial)
    assert walker.ot == pytest.approx(ovlp)
    eloc = walker.local_energy(system, trial)
    detR = walker.reortho(trial)
    eloc_new = walker.local_energy(system, trial)
    assert eloc == pytest.approx(eloc_new)
    assert detR*walker.ot == pytest.approx(ovlp)
