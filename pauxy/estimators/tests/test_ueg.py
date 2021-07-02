import numpy
import pytest
from pauxy.systems.ueg import UEG
from pauxy.estimators.ueg import fock_ueg, local_energy_ueg
from pauxy.estimators.greens_function import gab
from pauxy.utils.testing import get_random_wavefunction
from pauxy.utils.misc import timeit

@pytest.mark.unit
def test_fock_build():
    sys = UEG({'rs': 2.0, 'ecut': 2, 'nup': 7, 'ndown': 7, 'thermal': True})
    numpy.random.seed(7)
    psi = get_random_wavefunction(sys.nelec, sys.nbasis).real
    trial = numpy.eye(sys.nbasis, sys.nelec[0])
    G = numpy.array([gab(psi[:,:sys.nup], psi[:,:sys.nup]),
                     gab(psi[:,sys.nup:],
                         psi[:,sys.nup:])]).astype(numpy.complex128)
    nb = sys.nbasis
    # from pyscf import gto, scf, ao2mo
    # mol = gto.M()
    # mol.nelec = sys.nelec
    # mf = scf.UHF(mol)
    # U = sys.compute_real_transformation()
    # h1_8 = numpy.dot(U.conj().T, numpy.dot(sys.H1[0], U))
    # mf.get_hcore = lambda *args: h1_8
    # mf.get_ovlp = lambda *args: numpy.eye(nb)
    # mf._eri = sys.eri_8()
    # mf._eri = ao2mo.restore(8, eri_8, nb)
    # veff = mf.get_veff(dm=dm)
    eris = sys.eri_4()
    F = fock_ueg(sys, G)
    vj = numpy.einsum('pqrs,xqp->xrs', eris, G)
    vk = numpy.einsum('pqrs,xqr->xps', eris, G)
    fock = numpy.zeros((2,33,33), dtype=numpy.complex128)
    fock[0] = sys.H1[0] + vj[0] + vj[1] - vk[0]
    fock[1] = sys.H1[1] + vj[0] + vj[1] - vk[1]
    assert numpy.linalg.norm(fock - F) == pytest.approx(0.0)

@pytest.mark.unit
def test_build_J():
    sys = UEG({'rs': 2.0, 'ecut': 2.0, 'nup': 7, 'ndown': 7, 'thermal': True})
    Gkpq =  numpy.zeros((2,len(sys.qvecs)), dtype=numpy.complex128)
    Gpmq =  numpy.zeros((2,len(sys.qvecs)), dtype=numpy.complex128)
    psi = get_random_wavefunction(sys.nelec, sys.nbasis).real
    trial = numpy.eye(sys.nbasis, sys.nelec[0])
    G = numpy.array([gab(psi[:,:sys.nup], psi[:,:sys.nup]),
                     gab(psi[:,sys.nup:], psi[:,sys.nup:])])
    from pauxy.estimators.ueg import coulomb_greens_function
    for s in [0,1]:
        coulomb_greens_function(len(sys.qvecs), sys.ikpq_i,
                sys.ikpq_kpq, sys.ipmq_i, sys.ipmq_pmq, Gkpq[s],
                Gpmq[s], G[s])

    from pauxy.estimators.ueg import build_J
    J1 = timeit(build_J)(sys, Gpmq, Gkpq)
    from pauxy.estimators.ueg_kernels import build_J_opt
    J2 = timeit(build_J_opt)(len(sys.qvecs), sys.vqvec, sys.vol, sys.nbasis,
                        sys.ikpq_i, sys.ikpq_kpq, sys.ipmq_i, sys.ipmq_pmq,
                        Gkpq, Gpmq)
    assert numpy.linalg.norm(J1-J2) == pytest.approx(0.0)

@pytest.mark.unit
def test_build_K():
    sys = UEG({'rs': 2.0, 'ecut': 2.0, 'nup': 7, 'ndown': 7, 'thermal': True})
    Gkpq =  numpy.zeros((2,len(sys.qvecs)), dtype=numpy.complex128)
    Gpmq =  numpy.zeros((2,len(sys.qvecs)), dtype=numpy.complex128)
    psi = get_random_wavefunction(sys.nelec, sys.nbasis).real
    trial = numpy.eye(sys.nbasis, sys.nelec[0])
    G = numpy.array([gab(psi[:,:sys.nup], psi[:,:sys.nup]),
                     gab(psi[:,sys.nup:],
                         psi[:,sys.nup:])]).astype(numpy.complex128)
    from pauxy.estimators.ueg import build_K
    from pauxy.estimators.ueg_kernels import build_K_opt
    K1 = timeit(build_K)(sys, G)
    K2 = timeit(build_K_opt)(len(sys.qvecs), sys.vqvec, sys.vol, sys.nbasis,
                        sys.ikpq_i, sys.ikpq_kpq, sys.ipmq_i, sys.ipmq_pmq,
                        G)
    assert numpy.linalg.norm(K1-K2) == pytest.approx(0.0)
