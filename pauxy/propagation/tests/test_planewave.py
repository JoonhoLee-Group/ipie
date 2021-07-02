import numpy
import os
import pytest
from pauxy.systems.ueg import UEG
from pauxy.propagation.planewave import PlaneWave
from pauxy.walkers.single_det import SingleDetWalker
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.utils.misc import dotdict


@pytest.mark.unit
def test_pw():
    options = {'rs': 2, 'nup': 7, 'ndown': 7, 'ecut': 2,
               'write_integrals': True}
    system = UEG(inputs=options)
    occ = numpy.eye(system.nbasis)[:,:system.nup]
    wfn = numpy.zeros((1,system.nbasis,system.nup+system.ndown),
                      dtype=numpy.complex128)
    wfn[0,:,:system.nup] = occ
    wfn[0,:,system.nup:] = occ
    coeffs = numpy.array([1+0j])
    trial = MultiSlater(system, (coeffs, wfn))
    trial.psi = trial.psi[0]
    qmc = dotdict({'dt': 0.005, 'nstblz': 5})
    prop = PlaneWave(system, trial, qmc)
    walker = SingleDetWalker(system, trial)
    numpy.random.seed(7)
    a = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
    b = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
    wfn = (a + 1j*b).reshape((system.nbasis,system.nup+system.ndown))
    walker.phi = wfn.copy()
    walker.greens_function(trial)
    # fb = prop.construct_force_bias_slow(system, walker, trial)
    fb = prop.construct_force_bias(system, walker, trial)
    assert numpy.linalg.norm(fb) == pytest.approx(0.16660828645573392)
    xi = numpy.random.rand(system.nfields)
    vhs = prop.construct_VHS(system, xi-fb)
    assert numpy.linalg.norm(vhs) == pytest.approx(0.1467322554815581)

def teardown_module():
    cwd = os.getcwd()
    files = ['hamil.h5']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass
