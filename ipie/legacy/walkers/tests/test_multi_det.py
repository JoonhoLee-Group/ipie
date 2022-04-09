import itertools
import numpy
import os
import pytest
from ipie.systems.generic import Generic
from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.legacy.estimators.ci import simple_fci
from ipie.legacy.estimators.local_energy import local_energy, local_energy_generic_cholesky
from ipie.utils.misc import dotdict
from ipie.utils.linalg import reortho
from ipie.utils.testing import (
        generate_hamiltonian,
        get_random_wavefunction
        )
from ipie.legacy.walkers.multi_det import MultiDetWalker

@pytest.mark.unit
def test_walker_overlap():
    system = dotdict({'nup': 5, 'ndown': 5,
                      'nelec': (5,5), 'ne': 10})
    ham = dotdict({'nbasis':10})
    numpy.random.seed(7)
    a = numpy.random.rand(3*ham.nbasis*(system.nup+system.ndown))
    b = numpy.random.rand(3*ham.nbasis*(system.nup+system.ndown))
    wfn = (a + 1j*b).reshape((3,ham.nbasis,system.nup+system.ndown))
    coeffs = numpy.array([0.5+0j,0.3+0j,0.1+0j])
    trial = MultiSlater(system, ham, (coeffs, wfn))
    walker = MultiDetWalker(system, ham, trial)
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
    orbs = numpy.arange(ham.nbasis)
    oa = [c for c in itertools.combinations(orbs, system.nup)]
    ob = [c for c in itertools.combinations(orbs, system.ndown)]
    oa, ob = zip(*itertools.product(oa,ob))
    oa = oa[:5]
    ob = ob[:5]
    coeffs = numpy.array([0.9, 0.01, 0.01, 0.02, 0.04],
                         dtype=numpy.complex128)
    wfn = (coeffs,oa,ob)
    a = numpy.random.rand(ham.nbasis*(system.nup+system.ndown))
    b = numpy.random.rand(ham.nbasis*(system.nup+system.ndown))
    init = (a + 1j*b).reshape((ham.nbasis,system.nup+system.ndown))
    trial = MultiSlater(system, ham, wfn, init=init)
    walker = MultiDetWalker(system, ham, trial)
    I = numpy.eye(ham.nbasis)
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
        walk_ovlp = walker.ovlpsa[idet] * walker.ovlpsb[idet]
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
    system = Generic(nelec=nelec)
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((-1,nmo*nmo)).T.copy(),
                     ecore=enuc)
    (e0, ev), (d,oa,ob) = simple_fci(system, ham, gen_dets=True)
    na = system.nup
    init = get_random_wavefunction(nelec, nmo)
    init[:,:na], R = reortho(init[:,:na])
    init[:,na:], R = reortho(init[:,na:])
    trial = MultiSlater(system, ham, (ev[:,0],oa,ob), init=init)
    trial.calculate_energy(system, ham)
    walker = MultiDetWalker(system, ham, trial)
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
        # e = local_energy_G(system, numpy.array([ga,gb]))[0]
        e = local_energy_generic_cholesky(system, ham, numpy.array([ga,gb]))[0]
        nume += trial.coeffs[i].conj()*ovlp*e
        deno += trial.coeffs[i].conj()*ovlp
    e = local_energy(system, ham, walker, trial)[0]
    assert nume/deno == pytest.approx(e)
