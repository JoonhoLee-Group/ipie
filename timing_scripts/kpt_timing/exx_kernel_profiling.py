import numpy
from numba import jit
import math
from line_profiler import LineProfiler
import time

nk = 27
nchol = 250
nbsf = 26
nocc = 4
nwalkers = 10


def prof_kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset):
    exx = kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset)


@jit(nopython=True, fastmath=True)
def kpt_symmchol_exx_kernel_lowmem(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    kpq_mat : :class:`numpy.ndarray`
        all k + q in fractional coordinates.
    mq_vec : :class:`numpy.ndarray`
        all -q in fractional coordinates.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (naux, nk, nocc, nk, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    naux = rchola.shape[0]
    nocc = rchola.shape[2]
    nk = rchola.shape[1]
    exx = zeros(nwalkers, dtype=numpy.complex128)
    GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2)
    GhalfaT = GhalfaT.transpose(1, 3, 0, 2, 4).copy()
    GhalfaT2 = Ghalfa_batch.transpose(1, 3, 0, 2, 4).copy()
    rcholaT = rchola.transpose(1, 3, 0, 2, 4).copy()
    rcholbaraT = rcholbara.transpose(1, 3, 0, 2, 4).copy()
    T1 = zeros((nocc, nocc), dtype=numpy.complex128)
    T2 = zeros((nocc, nocc), dtype=numpy.complex128)
    for iq in range(len(Qset)):
        iq_real = Qset[iq]
        for ik in range(nk):
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                ik_pq = kpq_mat[iq_real, ik]
                Lkq = rcholaT[ik, iq]
                Lbarkpq = rcholbaraT[ikprime, iq]
                for iw in range(nwalkers):
                    Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq, iw]
                    Ghalf_k_kp = GhalfaT2[ik, ikprime, iw]
                    for g in range(naux):
                        T1 = Lkq[g] @ Ghalf_kpq_kprpq
                        T2 = Ghalf_k_kp @ Lbarkpq[g]
                        for i in range(nocc):
                            for j in range(nocc):
                                exx[iw] += -T1[i, j] * T2[i, j]

    return exx


@jit(nopython=True, fastmath=True)
def kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset):
    """Compute coulomb contribution for complex rchol with RHF trial.

    Parameters
    ----------
    rchol : :class:`numpy.ndarray`
        Half-rotated cholesky.
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis
    kpq_mat : :class:`numpy.ndarray`
        all k + q in fractional coordinates.
    mq_vec : :class:`numpy.ndarray`
        all -q in fractional coordinates.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (naux, nk, nocc, nq, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    # naux = rchola.shape[0]
    # nocc = rchola.shape[2]
    # nk = rchola.shape[1]
    # nbsf = rchola.shape[-1]
    naux = rchola.shape[3]
    nocc = rchola.shape[2]
    nk = rchola.shape[0]
    nbsf = rchola.shape[-1]
    exx = zeros(nwalkers, dtype=numpy.complex128)
    GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2)  # (nw, nk(nbsf), nbsf, nk(nocc), nocc)
    GhalfaT = GhalfaT.transpose(1, 3, 0, 2, 4).copy()  # (nk(nbsf), nk(nocc), nw, nbsf, nocc)
    GhalfaT2 = Ghalfa_batch.transpose(1, 3, 0, 2, 4).copy()  # (nk(nocc), nk(nbsf), nw, nocc, nbsf)
    # rcholaT = rchola.transpose(1,3,0,2,4).copy() # nk, nq, naux, nocc, nbsf
    # rcholbaraT = rcholbara.transpose(1,3,0,2,4).copy() # nk, nq, naux, nbsf, nocc
    T1 = zeros((nocc, nocc), dtype=numpy.complex128)
    T2 = zeros((nocc, nocc), dtype=numpy.complex128)
    GhalfaT0 = GhalfaT.transpose(0, 1, 3, 4, 2).copy()  # (nk(nbsf), nk(nocc), nbsf, nocc, nw)
    # rchola0 = rcholaT.transpose(0, 1, 3, 2, 4).copy()
    # rcholbara0 = rcholbaraT.transpose(0, 1, 3, 2, 4).copy()
    for iq in range(len(Qset)):
        iq_real = Qset[iq]
        for ik in range(nk):
            for ikprime in range(nk):
                ikpr_pq = kpq_mat[iq_real, ikprime]
                ik_pq = kpq_mat[iq_real, ik]
                # Lkq = rcholaT[ik, iq].transpose(1, 0, 2).copy().reshape(naux * nocc, -1)
                # Lbarkpq = rcholbaraT[ikprime, iq].transpose(1, 0, 2).copy().reshape(-1, naux * nocc)
                Lkq = rchola[ik, iq].reshape(naux * nocc, -1)
                Lbarkpq = rcholbara[ikprime, iq].reshape(-1, naux * nocc)
                # Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq].transpose(1, 2, 0).copy().reshape(nbsf, nocc * nwalkers)
                Ghalf_kpq_kprpq = GhalfaT0[ik_pq, ikpr_pq].reshape(nbsf, nocc * nwalkers)
                Ghalf_k_kp = GhalfaT2[ik, ikprime].reshape(nwalkers * nocc, nbsf)
                # Ghalf_kpq_kprpq = GhalfaT[ik_pq, ikpr_pq, iw]
                # Ghalf_k_kp = GhalfaT2[ik,ikprime, iw]
                # for g in range(naux):
                #     T1 = Lkq[g] @ Ghalf_kpq_kprpq
                #     T2 = Ghalf_k_kp @ Lbarkpq[g]
                #     # for i in range(nocc):
                #     #     for j in range(nocc):
                #     #         exx[iw] += - T1[i, j] * T2[i, j]
                T1 = Lkq @ Ghalf_kpq_kprpq  # (naux * nocc, nocc * nwalkers)
                T2 = Ghalf_k_kp @ Lbarkpq  # (nwalkers * nocc, naux * nocc)
                T1 = T1.reshape(naux * nocc * nocc, nwalkers).T.copy()
                T2 = T2.reshape(nwalkers, naux * nocc * nocc).copy()
                for iw in range(nwalkers):
                    exx[iw] += -T1[iw] @ T2[iw]
    return exx


@jit(nopython=True, fastmath=True)
def kpt_symmchol_exx_kernel2(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset):
    # sort out cupy later
    zeros = numpy.zeros
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (naux, nk, nocc, nq, nbsf) (gamma, k, i, q, p)
    # shape of Ghalf: (nw, nk, nocc, nk, nbsf)
    naux = rchola.shape[0]
    nocc = rchola.shape[2]
    nk = rchola.shape[1]
    exx = zeros(nwalkers, dtype=numpy.complex128)
    GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2).copy()  # (nw, nk(nbsf), nbsf, nk(nocc), nocc)
    GhalfaT2 = Ghalfa_batch.transpose(0, 3, 1, 2, 4).copy()  # (nw, nk(nbsf), nk(nocc), nocc, nbsf)

    rcholaT = rchola.transpose(3, 0, 1, 2, 4).copy()  # (nq, naux, nk, nocc, nbsf)
    rcholbaraT = rcholbara.transpose(3, 0, 1, 2, 4).copy()  # (nq, naux, nk, nbsf, nocc)

    T1 = zeros((nk, nocc, nk * nocc), dtype=numpy.complex128)
    T2 = zeros((nk, nk * nocc, nocc), dtype=numpy.complex128)
    for iw in range(nwalkers):
        Gw = GhalfaT[iw]  # (nk(nbsf), nbsf, nk(nocc), nocc)
        Gw2 = GhalfaT2[iw]  # (nk(nbsf), nk(nocc), nocc, nbsf)
        for iq in range(len(Qset)):
            iq_real = Qset[iq]
            for X in range(naux):
                LqX = rcholaT[iq, X]
                LbarqX = rcholbaraT[iq, X]
                ikpqs = kpq_mat[iq_real]  # all k+q is contained here
                Gw0 = Gw[:, :, ikpqs, :].copy()
                for ik in range(nk):
                    ikpq = ikpqs[ik]
                    Gkpq = Gw0[ikpq].reshape(nbsf, nk * nocc)
                    Gk = Gw2[ik].reshape(nocc * nk, nbsf)
                    T1[ik] = LqX[ik] @ Gkpq  # (nocc, nbsf) * (nbsf, nk*nocc)
                    T2[ik] = Gk @ LbarqX[ik]  # (nk*nocc,nbsf) (nbsf, nocc) -> (nk, nk*nocc, nocc)
                T3 = T2.transpose(0, 2, 1).copy()
                exx[iw] += -numpy.dot(T1.ravel(), T3.ravel())
    return exx


# kpq mat is a matrix with integers in the range of 0 to nk
kpq_mat = numpy.random.randint(0, nk, (nk, nk))

nq = nk // 2 + 1
Qset = numpy.arange(nq)

# rchola = numpy.random.rand(nchol, nk, nocc, nq, nbsf) + 1j * numpy.random.rand(nchol, nk, nocc, nq, nbsf)
# rcholbara = numpy.random.rand(nchol, nk, nbsf, nq, nocc) + 1j * numpy.random.rand(nchol, nk, nbsf, nq, nocc)
rchola = numpy.random.rand(nk, nq, nocc, nchol, nbsf) + 1j * numpy.random.rand(
    nk, nq, nocc, nchol, nbsf
)
rcholbara = numpy.random.rand(nk, nq, nbsf, nchol, nocc) + 1j * numpy.random.rand(
    nk, nq, nbsf, nchol, nocc
)


rchola_lowmem = rchola.transpose(3, 0, 2, 1, 4).copy()
rcholbara_lowmem = rcholbara.transpose(3, 0, 2, 1, 4).copy()
Ghalfa_batch = numpy.random.rand(nwalkers, nk, nocc, nk, nbsf) + 1j * numpy.random.rand(
    nwalkers, nk, nocc, nk, nbsf
)

# lp = LineProfiler()
exx = kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset)
# exx2 = kpt_symmchol_exx_kernel2(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset)
exx_lowmem = kpt_symmchol_exx_kernel_lowmem(
    rchola_lowmem, rcholbara_lowmem, Ghalfa_batch, kpq_mat, Qset
)
# profiled_fn = lp(prof_kpt_symmchol_exx_kernel)
# profiled_fn = lp(kpt_symmchol_exx_kernel)
time1 = time.time()
exx = kpt_symmchol_exx_kernel(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset)
time2 = time.time()
# exx2 = kpt_symmchol_exx_kernel2(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset)
# time3 = time.time()
exx_lowmem = kpt_symmchol_exx_kernel_lowmem(
    rchola_lowmem, rcholbara_lowmem, Ghalfa_batch, kpq_mat, Qset
)
time4 = time.time()
print(exx)
# print(exx2)
print(exx_lowmem)
print(f"elapsed time: {time2 - time1}")
# print(f"elapsed time2: {time3 - time2}")
print(f"elapsed time lowmem: {time4 - time2}")
# profiled_fn(rchola, rcholbara, Ghalfa_batch, kpq_mat, Qset)
# lp.print_stats()
