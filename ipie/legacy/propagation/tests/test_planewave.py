import os

import numpy
import pytest

from ipie.legacy.hamiltonians.ueg import UEG as HamUEG
from ipie.legacy.propagation.planewave import PlaneWave
from ipie.legacy.systems.ueg import UEG
from ipie.legacy.trial_wavefunction.multi_slater import MultiSlater
from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.utils.misc import dotdict


@pytest.mark.unit
def test_pw():
    options = {"rs": 2, "nup": 7, "ndown": 7, "ecut": 2, "write_integrals": True}
    system = UEG(options=options)
    ham = HamUEG(system, options=options)
    occ = numpy.eye(ham.nbasis)[:, : system.nup]
    wfn = numpy.zeros(
        (1, ham.nbasis, system.nup + system.ndown), dtype=numpy.complex128
    )
    wfn[0, :, : system.nup] = occ
    wfn[0, :, system.nup :] = occ
    coeffs = numpy.array([1 + 0j])
    trial = MultiSlater(system, ham, (coeffs, wfn))
    trial.psi = trial.psi[0]
    qmc = dotdict({"dt": 0.005, "nstblz": 5})
    prop = PlaneWave(system, ham, trial, qmc)
    walker = SingleDetWalker(system, ham, trial)
    numpy.random.seed(7)
    a = numpy.random.rand(ham.nbasis * (system.nup + system.ndown))
    b = numpy.random.rand(ham.nbasis * (system.nup + system.ndown))
    wfn = (a + 1j * b).reshape((ham.nbasis, system.nup + system.ndown))
    walker.phi = wfn.copy()
    walker.greens_function(trial)
    # fb = prop.construct_force_bias_slow(system, walker, trial)
    fb = prop.construct_force_bias(ham, walker, trial)
    assert numpy.linalg.norm(fb) == pytest.approx(0.16660828645573392)
    xi = numpy.random.rand(ham.nfields)
    vhs = prop.construct_VHS(ham, xi - fb)
    assert numpy.linalg.norm(vhs) == pytest.approx(0.1467322554815581)


def teardown_module():
    cwd = os.getcwd()
    files = ["hamil.h5"]
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass
