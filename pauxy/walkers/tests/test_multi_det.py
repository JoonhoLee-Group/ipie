import itertools
import numpy
import os
import pytest
from pauxy.systems.generic import Generic
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.estimators.ci import simple_fci
from pauxy.estimators.mixed import local_energy
from pauxy.utils.misc import dotdict
from pauxy.utils.linalg import reortho
from pauxy.utils.testing import (
        generate_hamiltonian,
        get_random_wavefunction
        )
from pauxy.walkers.multi_det import MultiDetWalker

@pytest.mark.unit
def test_walker_overlap():
    system = dotdict({'nup': 5, 'ndown': 5, 'nbasis': 10,
                      'nelec': (5,5), 'ne': 10})
    numpy.random.seed(7)
    a = numpy.random.rand(3*system.nbasis*(system.nup+system.ndown))
    b = numpy.random.rand(3*system.nbasis*(system.nup+system.ndown))
    wfn = (a + 1j*b).reshape((3,system.nbasis,system.nup+system.ndown))
    coeffs = numpy.array([0.5+0j,0.3+0j,0.1+0j])
    trial = MultiSlater(system, (coeffs, wfn))
    walker = MultiDetWalker(system, trial)
    def calc_ovlp(a,b):
        return numpy.linalg.det(numpy.dot(a.conj().T, b))
    ovlp = 0.0+0j
    na = system.nup
    pa = trial.psi[0,:,:na]
    pb = trial.psi[0,:,na:]
    for i, d in enumerate(trial.psi):
        ovlp += coeffs[i].conj()*calc_ovlp(d[:,:na],pa)*calc_ovlp(d[:,na:],pb)
    assert ovlp.real == pytest.approx(walker.ovlp.real)
    assert ovlp.imag == pytest.approx(walker.ovlp.imag)
    # Test PH type wavefunction.
    orbs = numpy.arange(system.nbasis)
    oa = [c for c in itertools.combinations(orbs, system.nup)]
    ob = [c for c in itertools.combinations(orbs, system.ndown)]
    oa, ob = zip(*itertools.product(oa,ob))
    oa = oa[:5]
    ob = ob[:5]
    coeffs = numpy.array([0.9, 0.01, 0.01, 0.02, 0.04],
                         dtype=numpy.complex128)
    wfn = (coeffs,oa,ob)
    a = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
    b = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
    init = (a + 1j*b).reshape((system.nbasis,system.nup+system.ndown))
    trial = MultiSlater(system, wfn, init=init)
    walker = MultiDetWalker(system, trial)
    I = numpy.eye(system.nbasis)
    ovlp_sum = 0.0
    for idet, (c, occa, occb) in enumerate(zip(coeffs,oa,ob)):
        psia = I[:,occa]
        psib = I[:,occb]
        sa = numpy.dot(psia.conj().T, init[:,:system.nup])
        sb = numpy.dot(psib.conj().T, init[:,system.nup:])
        isa = numpy.linalg.inv(sa)
        isb = numpy.linalg.inv(sb)
        ga = numpy.dot(init[:,:system.nup], numpy.dot(isa, psia.conj().T)).T
        gb = numpy.dot(init[:,system.nup:], numpy.dot(isb, psib.conj().T)).T
        ovlp = numpy.linalg.det(sa)*numpy.linalg.det(sb)
        ovlp_sum += c.conj()*ovlp
        walk_ovlp = walker.ovlps[idet]
        assert ovlp == pytest.approx(walk_ovlp)
        assert numpy.linalg.norm(ga-walker.Gi[idet,0]) == pytest.approx(0)
        assert numpy.linalg.norm(gb-walker.Gi[idet,1]) == pytest.approx(0)
    assert ovlp_sum == pytest.approx(walker.ovlp)

@pytest.mark.unit
def test_walker_energy():
    numpy.random.seed(7)
    nelec = (2,2)
    nmo = 5
    h1e, chol, enuc, eri = generate_hamiltonian(nmo, nelec, cplx=False)
    system = Generic(nelec=nelec, h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=enuc)
    (e0, ev), (d,oa,ob) = simple_fci(system, gen_dets=True)
    na = system.nup
    init = get_random_wavefunction(nelec, nmo)
    init[:,:na], R = reortho(init[:,:na])
    init[:,na:], R = reortho(init[:,na:])
    trial = MultiSlater(system, (ev[:,0],oa,ob), init=init)
    trial.calculate_energy(system)
    walker = MultiDetWalker(system, trial)
    nume = 0
    deno = 0
    for i in range(trial.ndets):
        psia = trial.psi[i,:,:na]
        psib = trial.psi[i,:,na:]
        oa = numpy.dot(psia.conj().T, init[:,:na])
        ob = numpy.dot(psib.conj().T, init[:,na:])
        isa = numpy.linalg.inv(oa)
        isb = numpy.linalg.inv(ob)
        ovlp = numpy.linalg.det(oa)*numpy.linalg.det(ob)
        ga = numpy.dot(init[:,:system.nup], numpy.dot(isa, psia.conj().T)).T
        gb = numpy.dot(init[:,system.nup:], numpy.dot(isb, psib.conj().T)).T
        e = local_energy(system, numpy.array([ga,gb]))[0]
        nume += trial.coeffs[i].conj()*ovlp*e
        deno += trial.coeffs[i].conj()*ovlp
    print(nume/deno,nume,deno,e0[0])
