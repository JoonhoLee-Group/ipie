import itertools
import numpy
from pyqumc.utils.misc import dotdict
from pyqumc.utils.linalg import modified_cholesky

def generate_hamiltonian_low_mem(nmo, nelec, cplx=False):
    h1e = numpy.random.random((nmo,nmo))
    if cplx:
        h1e = h1e + 1j*numpy.random.random((nmo,nmo))
    chol = numpy.random.rand(nmo**3 * 4).reshape((nmo*4,nmo,nmo))
    enuc = numpy.random.rand()
    return h1e, chol, enuc

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

def get_random_nomsd(nup, ndown, nbasis, ndet=10, cplx=True):
    a = numpy.random.rand(ndet*nbasis*(nup+ndown))
    b = numpy.random.rand(ndet*nbasis*(nup+ndown))
    if cplx:
        wfn = (a + 1j*b).reshape((ndet,nbasis,nup+ndown))
        coeffs = numpy.random.rand(ndet)+1j*numpy.random.rand(ndet)
    else:
        wfn = a.reshape((ndet,nbasis,nup+ndown))
        coeffs = numpy.random.rand(ndet)
    return (coeffs,wfn)

def get_random_phmsd(nup, ndown, nbasis, ndet=10, init=False, shuffle = False):
    orbs = numpy.arange(nbasis)
    oa = [c for c in itertools.combinations(orbs, nup)]
    ob = [c for c in itertools.combinations(orbs, ndown)]
    oa, ob = zip(*itertools.product(oa,ob))

    if (shuffle):
        ntot = len(oa)
        det_list = [numpy.random.randint(0, ntot-1) for i in range(ndet)] # this may pick duplicated list...
        oa = numpy.array(oa)
        ob = numpy.array(ob)
        oa_new = oa[det_list,:]
        ob_new = ob[det_list,:]
        oa = oa_new.copy()
        ob = ob_new.copy()
    else:
        oa = oa[:ndet]
        ob = ob[:ndet]
    coeffs = numpy.random.rand(ndet)+1j*numpy.random.rand(ndet)
    wfn = (coeffs,oa,ob)
    if init:
        a = numpy.random.rand(nbasis*(nup+ndown))
        b = numpy.random.rand(nbasis*(nup+ndown))
        init_wfn = (a + 1j*b).reshape((nbasis,nup+ndown))
    return wfn, init_wfn

def get_random_wavefunction(nelec, nbasis):
    na = nelec[0]
    nb = nelec[1]
    a = numpy.random.rand(nbasis*(na+nb))
    b = numpy.random.rand(nbasis*(na+nb))
    init = (a + 1j*b).reshape((nbasis,na+nb))
    return init
