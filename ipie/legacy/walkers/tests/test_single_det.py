import numpy
import pytest

from ipie.legacy.estimators.greens_function import gab
from ipie.legacy.estimators.local_energy import local_energy
from ipie.legacy.hamiltonians.hubbard import Hubbard
from ipie.legacy.propagation.hubbard import Hirsch
from ipie.legacy.trial_wavefunction.multi_slater import MultiSlater
from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.systems.generic import Generic
from ipie.utils.misc import dotdict


@pytest.mark.unit
def test_overlap():
    options = {"nx": 4, "ny": 4, "nup": 8, "ndown": 8, "U": 4}
    system = Generic((8, 8), verbose=False)
    ham = Hubbard(options, verbose=False)
    eigs, eigv = numpy.linalg.eigh(ham.H1[0])
    coeffs = numpy.array([1.0 + 0j])
    wfn = numpy.zeros((1, ham.nbasis, system.ne))
    wfn[0, :, : system.nup] = eigv[:, : system.nup].copy()
    wfn[0, :, system.nup :] = eigv[:, : system.ndown].copy()
    trial = MultiSlater(system, ham, (coeffs, wfn))
    trial.psi = trial.psi[0]
    walker = SingleDetWalker(system, ham, trial)
    nup = system.nup
    # Test overlap
    ovlp = numpy.dot(trial.psi[:, :nup].conj().T, walker.phi[:, :nup])
    id_exp = numpy.dot(walker.inv_ovlp[0], ovlp)
    numpy.testing.assert_allclose(id_exp, numpy.eye(nup), atol=1e-12)


@pytest.mark.unit
def test_update_overlap():
    options = {"nx": 4, "ny": 4, "nup": 8, "ndown": 8, "U": 4}
    system = Generic((8, 8), verbose=False)
    ham = Hubbard(options, verbose=False)
    eigs, eigv = numpy.linalg.eigh(ham.H1[0])
    coeffs = numpy.array([1.0 + 0j])
    wfn = numpy.zeros((1, ham.nbasis, system.ne))
    wfn[0, :, : system.nup] = eigv[:, : system.nup].copy()
    wfn[0, :, system.nup :] = eigv[:, : system.ndown].copy()
    trial = MultiSlater(system, ham, (coeffs, wfn))
    trial.psi = trial.psi[0]
    walker = SingleDetWalker(system, ham, trial)
    # Test update
    nup = system.nup
    vtup = walker.phi[3, :nup] * 0.333
    walker.phi[3, :nup] = walker.phi[3, :nup] + vtup
    vtdn = walker.phi[3, nup:] * -0.333
    walker.phi[3, nup:] = walker.phi[3, nup:] + vtdn
    walker.update_inverse_overlap(trial, vtup, vtdn, 3)
    ovlp = numpy.dot(trial.psi[:, :nup].conj().T, walker.phi[:, :nup])
    id_exp = numpy.dot(walker.inv_ovlp[0], ovlp)
    numpy.testing.assert_allclose(id_exp, numpy.eye(nup), atol=1e-12)
    ovlp = numpy.dot(trial.psi[:, nup:].conj().T, walker.phi[:, nup:])
    id_exp = numpy.dot(walker.inv_ovlp[1], ovlp)
    numpy.testing.assert_allclose(id_exp, numpy.eye(nup), atol=1e-12)


@pytest.mark.unit
def test_greens_function():
    options = {"nx": 4, "ny": 4, "nup": 8, "ndown": 8, "U": 4}
    system = Generic((8, 8), verbose=False)
    ham = Hubbard(options, verbose=False)
    eigs, eigv = numpy.linalg.eigh(ham.H1[0])
    coeffs = numpy.array([1.0 + 0j])
    wfn = numpy.zeros((1, ham.nbasis, system.ne))
    wfn[0, :, : system.nup] = eigv[:, : system.nup].copy()
    wfn[0, :, system.nup :] = eigv[:, : system.ndown].copy()
    trial = MultiSlater(system, ham, (coeffs, wfn))
    trial.psi = trial.psi[0]
    nup = system.nup
    walker = SingleDetWalker(system, ham, trial)
    # Test Green's function
    Gref = gab(trial.psi[:, :nup], walker.phi[:, :nup])
    numpy.testing.assert_allclose(walker.G[0], Gref, atol=1e-12)


@pytest.mark.unit
def test_reortho():
    options = {"nx": 4, "ny": 4, "nup": 8, "ndown": 8, "U": 4}
    system = Generic((8, 8), verbose=False)
    ham = Hubbard(options, verbose=False)
    eigs, eigv = numpy.linalg.eigh(ham.H1[0])
    coeffs = numpy.array([1.0 + 0j])
    numpy.random.seed(7)
    wfn = numpy.random.random((ham.nbasis * system.ne)).reshape(
        1, ham.nbasis, system.ne
    )
    trial = MultiSlater(system, ham, (coeffs, wfn))
    trial.psi = trial.psi[0]
    nup = system.nup
    walker = SingleDetWalker(system, ham, trial)
    # Test Green's function
    ovlp = walker.calc_overlap(trial)
    assert walker.ot == pytest.approx(ovlp)
    eloc = local_energy(system, ham, walker, trial)
    detR = walker.reortho(trial)
    eloc_new = local_energy(system, ham, walker, trial)
    assert eloc == pytest.approx(eloc_new)
    assert detR * walker.ot == pytest.approx(ovlp)
