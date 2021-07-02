import itertools
import numpy
from pauxy.utils.misc import dotdict
from pauxy.utils.linalg import modified_cholesky

def generate_hamiltonian(nmo, nelec, cplx=False, sym=8):
    h1e = numpy.random.random((nmo,nmo))
    if cplx:
        h1e = h1e + 1j*numpy.random.random((nmo,nmo))
    eri = numpy.random.normal(scale=0.01, size=(nmo,nmo,nmo,nmo))
    if cplx:
        eri = eri + 1j*numpy.random.normal(scale=0.01, size=(nmo,nmo,nmo,nmo))
    # Restore symmetry to the integrals.
    if sym >= 4:
        # (ik|jl) = (jl|ik)
        # (ik|jl) = (ki|lj)*
        eri = eri + eri.transpose(2,3,0,1)
        eri = eri + eri.transpose(3,2,1,0).conj()
    if sym == 8:
        eri = eri + eri.transpose(1,0,2,3)
    # Construct hermitian matrix M_{ik,lj}.
    eri = eri.transpose((0,1,3,2))
    eri = eri.reshape((nmo*nmo,nmo*nmo))
    # Make positive semi-definite.
    eri = numpy.dot(eri,eri.conj().T)
    chol = modified_cholesky(eri, tol=1e-3, verbose=False, cmax=30)
    chol = chol.reshape((-1,nmo,nmo))
    enuc = numpy.random.rand()
    return h1e, chol, enuc, eri

def get_random_nomsd(system, ndet=10, cplx=True):
    a = numpy.random.rand(ndet*system.nbasis*(system.nup+system.ndown))
    b = numpy.random.rand(ndet*system.nbasis*(system.nup+system.ndown))
    if cplx:
        wfn = (a + 1j*b).reshape((ndet,system.nbasis,system.nup+system.ndown))
        coeffs = numpy.random.rand(ndet)+1j*numpy.random.rand(ndet)
    else:
        wfn = a.reshape((ndet,system.nbasis,system.nup+system.ndown))
        coeffs = numpy.random.rand(ndet)
    return (coeffs,wfn)

def get_random_phmsd(system, ndet=10, init=False):
    orbs = numpy.arange(system.nbasis)
    oa = [c for c in itertools.combinations(orbs, system.nup)]
    ob = [c for c in itertools.combinations(orbs, system.ndown)]
    oa, ob = zip(*itertools.product(oa,ob))
    oa = oa[:ndet]
    ob = ob[:ndet]
    coeffs = numpy.random.rand(ndet)+1j*numpy.random.rand(ndet)
    wfn = (coeffs,oa,ob)
    if init:
        a = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
        b = numpy.random.rand(system.nbasis*(system.nup+system.ndown))
        init_wfn = (a + 1j*b).reshape((system.nbasis,system.nup+system.ndown))
    return wfn, init_wfn

def get_random_wavefunction(nelec, nbasis):
    na = nelec[0]
    nb = nelec[1]
    a = numpy.random.rand(nbasis*(na+nb))
    b = numpy.random.rand(nbasis*(na+nb))
    init = (a + 1j*b).reshape((nbasis,na+nb))
    return init
