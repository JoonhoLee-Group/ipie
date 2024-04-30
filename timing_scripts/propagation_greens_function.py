try:
    import cupy as np

    _gpu = True
except ImportError:
    import numpy as np

    _gpu = False
import time

from ipie.hamiltonians.generic import Generic as HamGeneric
from ipie.legacy.estimators.greens_function import (
    greens_function_single_det,
    greens_function_single_det_batch,
)
from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.legacy.walkers.single_det_batch import SingleDetWalkerBatch
from ipie.systems.generic import Generic
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.utils.testing import get_random_nomsd

divide = 5

nao = 1000 // divide
nocc = 200 // divide
naux = 4000 // divide
nwalkers = 20


def time_overlap():
    def loop_based(a, b):
        for iw in range(a.shape[0]):
            # print(a.shape, b.shape)
            ovlp = np.dot(a[iw].T, b.conj())
            # print(ovlp.shape)
            inv = np.linalg.inv(ovlp)
            s, o = np.linalg.slogdet(inv)

    def einsum_based(a, b):
        ovlps = np.einsum("wmi,mj->wij", a, b.conj(), optimize=True)
        invs = np.linalg.inv(ovlps)
        s, o = np.linalg.slogdet(invs)

    # Ghalf construction
    for nwalkers in range(1, 40, 5):
        psi = np.random.random((nwalkers, nao, nocc))
        trial = np.random.random((nao, nocc))
        start = time.time()
        loop_based(psi, trial)
        t_loop = time.time() - start
        start = time.time()
        einsum_based(psi, trial)
        t_einsum = time.time() - start
        print(nwalkers, t_einsum / t_loop)


def time_dets():
    def loop_based(ovlp):
        for iw in range(ovlp.shape[0]):
            inv = np.linalg.inv(ovlp)
            s, o = np.linalg.slogdet(inv)

    def einsum_based(ovlp):
        invs = np.linalg.inv(ovlps)
        s, o = np.linalg.slogdet(invs)

    # Ghalf construction
    for nwalkers in range(1, 40, 5):
        ovlps = np.random.random((nwalkers, nocc, nocc))
        start = time.time()
        loop_based(ovlps)
        t_loop = time.time() - start
        start = time.time()
        einsum_based(ovlps)
        t_einsum = time.time() - start
        print(nwalkers, t_einsum / t_loop)


def time_ghalf():
    def loop_based(a, b, out):
        for iw in range(a.shape[0]):
            out[iw] = np.dot(b[iw], a[iw].T)

    def einsum_based(a, b, out):
        out = np.einsum("wij,wmj->wim", b, a, optimize=True)

    def dot_based(a, b, out):
        nw = a.shape[0]
        no = b.shape[1]
        nb = a.shape[1]
        a_ = a.reshape((nw * nb, no))
        b_ = b.reshape((nw * no, no))
        out = np.dot(b_, a_.T)

    # Ghalf construction
    for nwalkers in range(1, 40, 5):
        walkers = np.random.random((nwalkers, nao, nocc))
        gf = np.random.random((nwalkers, nocc, nao))
        ovlps = np.random.random((nwalkers, nocc, nocc))
        start = time.time()
        loop_based(walkers, ovlps, gf)
        t_loop = time.time() - start
        start = time.time()
        einsum_based(walkers, ovlps, gf)
        t_einsum = time.time() - start
        start = time.time()
        dot_based(walkers, ovlps, gf)
        t_dot = time.time() - start
        print(nwalkers, t_einsum / t_loop, t_dot / t_loop)


def time_gfull():
    def loop_based(a, b, out):
        for iw in range(a.shape[0]):
            out[iw] = np.dot(b.conj(), a[iw])

    def einsum_based(a, b, out):
        out = np.einsum("mi,win->wmn", b.conj(), a, optimize=True)

    # Ghalf construction
    for nwalkers in range(1, 40, 5):
        trial = np.random.random((nao, nocc))
        ghalf = np.random.random((nwalkers, nocc, nao))
        gf = np.random.random((nwalkers, nao, nao))
        ovlps = np.random.random((nwalkers, nocc, nocc))
        start = time.time()
        loop_based(ghalf, trial, gf)
        t_loop = time.time() - start
        start = time.time()
        einsum_based(ghalf, trial, gf)
        t_einsum = time.time() - start
        print(nwalkers, t_einsum / t_loop)


# Full GF test
def time_routines():
    for nwalkers in range(1, 40, 5):
        wfn = get_random_nomsd(nocc, nocc, nao, ndet=1)
        h1e = np.random.random((nao, nao))
        system = Generic(nelec=(nocc, nocc))
        nmo = nao
        chol = np.zeros((naux, nmo, nmo))
        ham = HamGeneric(
            h1e=np.array([h1e, h1e]), chol=chol.reshape((naux, nmo * nmo)).T.copy(), ecore=0
        )
        if _gpu:
            ham.cast_to_cupy()
        trial = MultiSlater(system, ham, wfn)
        trial.psia = trial.psi[0, :, :nocc].copy()
        trial.psib = trial.psi[0, :, nocc:].copy()
        trial.psi = trial.psi[0]
        walkers = [SingleDetWalker(system, ham, trial) for _ in range(nwalkers)]
        walker_batch = SingleDetWalkerBatch(system, ham, trial, nwalkers)
        start = time.time()
        greens_function_single_det(walker_batch, trial)
        loop = time.time() - start
        start = time.time()
        greens_function_single_det_batch(walker_batch, trial)
        print(nwalkers, (time.time() - start) / loop)


if __name__ == "__main__":
    tmp = np.dot(np.random.random((100, 100)), np.eye(100))
    print(">>>> Overlap <<<<<")
    time_overlap()
    print(">>>> Dets <<<<<")
    time_dets()
    print(">>>> Ghalf <<<<<")
    time_ghalf()
    print(">>>> Gfull <<<<<")
    time_gfull()
    print(">>>> Actual Routines <<<<<")
    time_routines()
