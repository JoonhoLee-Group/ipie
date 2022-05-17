import itertools
import numpy
from ipie.utils.misc import dotdict
from ipie.utils.linalg import modified_cholesky

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

def get_random_phmsd_opt(nup, ndown, nbasis, ndet=10, init=False):
    pass
    # orbs = numpy.arange(nbasis)
    # coeffs = numpy.random.rand(ndet)+1j*numpy.random.rand(ndet)
    # # Put in HF det
    # assert nup == ndown
    # aufbau_a = numpy.arange(nup)
    # aufbau_b = numpy.arange(ndown)
    # nex_a = numpy.zeros((ndet, nup), dtype=numpy.int32)
    # nex_b = numpy.zeros((ndet, ndown), dtype=numpy.int32)
    # occ_a[0] = aufbau_a
    # occ_b[0] = aufbau_b
    # ndet_per_level = ndet // (nup+ndown)
    # for nex in range(nup+ndown):
        # start = nex * ndet_per_level
        # end   = min((nex+1) * ndet_per_level, ndet)
        # for idet in range(start, end)
            # non_unique = 0
            # found_unique = False
            # while not found_unique:
                # new_det = numpy.sort(numpy.random.choice(nbasis, nup, replace=False))
                # if (new_det == occ_a[:idet]).all(1).any():
                    # non_unique += 1
                # else:
                    # occ_a[idet] = new_det
                    # found_unique = True
                # if non_unique > 10:
                    # occ_a[idet] = new_det
                    # break
            # non_unique = 0
            # found_unique = False
            # while not found_unique:
                # new_det = numpy.sort(numpy.random.choice(nbasis, ndown, replace=False))
                # if (new_det == occ_a[:idet]).all(1).any():
                    # non_unique += 1
                # else:
                    # occ_b[idet] = new_det
                    # found_unique = True
                # if non_unique > 10:
                    # break
    # wfn = (coeffs, occ_a, occ_b)
    # if init:
        # a = numpy.random.rand(nbasis*(nup+ndown))
        # b = numpy.random.rand(nbasis*(nup+ndown))
        # init_wfn = (a + 1j*b).reshape((nbasis,nup+ndown))
    # return wfn, init_wfn

def get_random_wavefunction(nelec, nbasis):
    na = nelec[0]
    nb = nelec[1]
    a = numpy.random.rand(nbasis*(na+nb))
    b = numpy.random.rand(nbasis*(na+nb))
    init = (a + 1j*b).reshape((nbasis,na+nb))
    return init

def generate_hamiltonian_low_mem(nmo, nelec, cplx=False):
    h1e = numpy.random.random((nmo,nmo))
    if cplx:
        h1e = h1e + 1j*numpy.random.random((nmo,nmo))
    chol = numpy.random.rand(nmo**3 * 4).reshape((nmo*4,nmo,nmo))
    enuc = numpy.random.rand()
    return h1e, chol, enuc

def shaped_normal(shape, cmplx=False):
    size = numpy.prod(shape)
    if cmplx:
        arr_r = numpy.random.normal(size=size)
        arr_i = numpy.random.normal(size=size)
        arr = arr_r + 1j*arr_i
    else:
        arr = numpy.random.normal(size=size)
    return arr.reshape(shape)

def gen_random_test_instances(nmo, nocc, naux, nwalkers, seed=7):
    numpy.random.seed(seed)
    wfn = get_random_nomsd(nocc, nocc, nmo, ndet=1)
    h1e = shaped_normal((nmo,nmo))
    from ipie.systems import Generic
    system = Generic(nelec=(nocc,nocc))
    chol = shaped_normal((naux,nmo,nmo))
    from ipie.hamiltonians import Generic as HamGeneric
    ham = HamGeneric(h1e=numpy.array([h1e,h1e]),
                     chol=chol.reshape((naux,nmo*nmo)).T.copy(),
                     h1e_mod=h1e.copy(),
                     ecore=0, verbose=False)
    from ipie.trial_wavefunction import MultiSlater
    trial = MultiSlater(system, ham, wfn,
            options={'build_greens_function': False})
    trial.psia = trial.psi[0,:,:nocc].copy()
    trial.psib = trial.psi[0,:,nocc:].copy()
    from ipie.walkers import SingleDetWalkerBatch
    walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)
    Ghalfa = shaped_normal((nwalkers, nocc, nmo), cmplx=True)
    Ghalfb = shaped_normal((nwalkers, nocc, nmo), cmplx=True)
    trial._rchola = shaped_normal((naux, nocc*nmo))
    trial._rcholb = shaped_normal((naux, nocc*nmo))
    trial._rH1a = shaped_normal((nocc, nmo))
    trial._rH1b = shaped_normal((nocc, nmo))
    trial.psi = trial.psi[0]
    return system, ham, walker_batch, trial
