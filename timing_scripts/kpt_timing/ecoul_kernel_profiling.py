import numpy
from numba import jit
import math
from line_profiler import LineProfiler

nk = 54
nchol = 200
nbsf = 52
nocc = 4
nwalkers = 16


def prof_kpt_symmchol_ecoul_kernel_uhf(
    rcholaT, rcholbT, rcholbaraT, rcholbarbT, GhalfaT, GhalfbT, GhalfaT2, GhalfbT2, kpq_mat, Qset
):
    # ecoul = kpt_symmchol_ecoul_kernel_uhf(rcholaT, rcholbT, rcholbaraT, rcholbarbT, GhalfaT, GhalfbT, GhalfaT2, GhalfbT2, kpq_mat, Qset)
    ecoul = kpt_symmchol_ecoul_kernel_uhf_opt(
        rcholaT,
        rcholbT,
        rcholbaraT,
        rcholbarbT,
        GhalfaT,
        GhalfbT,
        GhalfaT2,
        GhalfbT2,
        kpq_mat,
        Qset,
    )


@jit(nopython=True, fastmath=True)
def kpt_symmchol_ecoul_kernel_uhf(
    rchola, rcholb, rcholbara, rcholbarb, Ghalfa, Ghalfb, GhalfaT, GhalfbT, kpq_mat, Qset
):
    """Compute coulomb contribution for real rchol with UHF trial.

    Parameters
    ----------
    rchola : :class:`numpy.ndarray`
        Half-rotated cholesky (alpha).
    rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky (beta).
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis.
    Ghalfb : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nbeta x nbasis.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    multiply = numpy.multiply
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (nq, nk, naux, nocc, nbsf) (q, k, gamma, i, p)
    # shape of Ghalf: (nk, nk, nw, nocc, nbsf) (k1, k2, w, i, p)
    naux = rchola.shape[2]
    nk = rchola.shape[1]
    nq = rchola.shape[0]
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = zeros((nwalkers, naux, nq), dtype=numpy.complex128)
    Xbar = zeros((nwalkers, naux, nq), dtype=numpy.complex128)
    for iq in range(len(Qset)):
        iq_real = Qset[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            for iw in range(nwalkers):
                Ghalfa_k_kpq = Ghalfa[ik, ik_pq, iw]
                GhalfTa_k_kpq = GhalfaT[ik, ik_pq, iw]
                Ghalfb_k_kpq = Ghalfb[ik, ik_pq, iw]
                GhalfTb_k_kpq = GhalfbT[ik, ik_pq, iw]
                La = rchola[iq, ik]
                Lb = rcholb[iq, ik]
                Lbara = rcholbara[iq, ik]
                Lbarb = rcholbarb[iq, ik]
                for g in range(naux):
                    # X[iw, g, iq] += numpy.trace(dot(rchola[g, ik, :, iq, :], GhalfaT[iw, ik_pq, :, ik, :])) + numpy.trace(dot(rcholb[g, ik, :, iq, :], GhalfbT[iw, ik_pq, :, ik, :]))
                    # X[iw, g, iq] += numpy.sum(multiply(La[g], Ghalfa_k_kpq)) + numpy.sum(multiply(Lb[g], Ghalfb_k_kpq))
                    # Xbar[iw, g, iq] += numpy.sum(multiply(rcholbara[g, ik, :, iq, :], GhalfaT[iw, ik, :, ik_pq, :])) + numpy.sum(multiply(rcholbarb[g, ik, :, iq, :], GhalfbT[iw, ik, :, ik_pq, :]))
                    # Xbar[iw, g, iq] += numpy.sum(multiply(Lbara[g], GhalfTa_k_kpq)) + numpy.sum(multiply(Lbarb[g], GhalfTb_k_kpq))
                    for i in range(nocc):
                        for mu in range(nbsf):
                            X[iw, g, iq] += (
                                La[g, i, mu] * Ghalfa_k_kpq[i, mu]
                                + Lb[g, i, mu] * Ghalfb_k_kpq[i, mu]
                            )
                            Xbar[iw, g, iq] += (
                                Lbara[g, i, mu] * GhalfTa_k_kpq[i, mu]
                                + Lbarb[g, i, mu] * GhalfTb_k_kpq[i, mu]
                            )

    X = X.reshape(nwalkers, naux * nq).copy()
    Xbar = Xbar.reshape(nwalkers, naux * nq).copy()
    for iw in range(nwalkers):
        ecoul[iw] = dot(X[iw], Xbar[iw])
    return 0.5 * ecoul / nk


def kpt_symmchol_ecoul_kernel_uhf_opt(
    rchola, rcholb, rcholbara, rcholbarb, Ghalfa, Ghalfb, GhalfaT, GhalfbT, kpq_mat, Qset
):
    """Compute coulomb contribution for real rchol with UHF trial.

    Parameters
    ----------
    rchola : :class:`numpy.ndarray`
        Half-rotated cholesky (alpha).
    rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky (beta).
    Ghalfa : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nalpha  x nbasis.
    Ghalfb : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nbeta x nbasis.

    Returns
    -------
    ecoul : :class:`numpy.ndarray`
        coulomb contribution for all walkers.
    """
    # sort out cupy later
    zeros = numpy.zeros
    dot = numpy.dot
    multiply = numpy.multiply
    nwalkers = Ghalfa_batch.shape[0]

    # shape of rchola: (nq, nk, naux, nocc, nbsf) (q, k, gamma, i, p)
    # shape of Ghalf: (nk, nk, nwalkers, nocc, nbsf) (k1, k2, w, i, p)
    nbsf = rchola.shape[4]
    naux = rchola.shape[2]
    nk = rchola.shape[1]
    ecoul = zeros(nwalkers, dtype=numpy.complex128)
    X = zeros((len(Qset), nwalkers, naux), dtype=numpy.complex128)
    Xbar = zeros((len(Qset), nwalkers, naux), dtype=numpy.complex128)
    for iq in range(len(Qset)):
        iq_real = Qset[iq]
        Xq = X[iq]
        Xbarq = Xbar[iq]
        for ik in range(nk):
            ik_pq = kpq_mat[iq_real, ik]
            La = rchola[iq, ik].reshape(naux, nocc * nbsf)
            Lb = rcholb[iq, ik].reshape(naux, nocc * nbsf)
            Lbara = rcholbara[iq, ik].reshape(naux, nocc * nbsf)
            Lbarb = rcholbarb[iq, ik].reshape(naux, nocc * nbsf)
            for iw in range(nwalkers):
                Ghalfa_k_kpq = Ghalfa[ik, ik_pq, iw].reshape(nocc * nbsf)
                GhalfTa_k_kpq = GhalfaT[ik, ik_pq, iw].reshape(nocc * nbsf)
                Ghalfb_k_kpq = Ghalfb[ik, ik_pq, iw].reshape(nocc * nbsf)
                GhalfTb_k_kpq = GhalfbT[ik, ik_pq, iw].reshape(nocc * nbsf)
                Xq[iw] += La @ Ghalfa_k_kpq + Lb @ Ghalfb_k_kpq
                Xbarq[iw] += Lbara @ GhalfTa_k_kpq + Lbarb @ GhalfTb_k_kpq
    X = X.transpose(1, 0, 2).copy()
    Xbar = Xbar.transpose(1, 0, 2).copy()
    X = X.reshape(nwalkers, naux * len(Qset))
    Xbar = Xbar.reshape(nwalkers, naux * len(Qset))
    for iw in range(nwalkers):
        ecoul[iw] = dot(X[iw], Xbar[iw])
    return 0.5 * ecoul / nk


nq = nk // 2 + 1
Qset = numpy.arange(nq)

# Initialize the matrices
rchola = numpy.random.rand(nchol, nk, nocc, nq, nbsf) + 1j * numpy.random.rand(
    nchol, nk, nocc, nq, nbsf
)
rcholb = numpy.random.rand(nchol, nk, nocc, nq, nbsf) + 1j * numpy.random.rand(
    nchol, nk, nocc, nq, nbsf
)
rcholbara = numpy.random.rand(nchol, nk, nbsf, nq, nocc) + 1j * numpy.random.rand(
    nchol, nk, nbsf, nq, nocc
)
rcholbarb = numpy.random.rand(nchol, nk, nbsf, nq, nocc) + 1j * numpy.random.rand(
    nchol, nk, nbsf, nq, nocc
)

Ghalfa_batch = numpy.random.rand(nwalkers, nk, nocc, nk, nbsf) + 1j * numpy.random.rand(
    nwalkers, nk, nocc, nk, nbsf
)
Ghalfb_batch = numpy.random.rand(nwalkers, nk, nocc, nk, nbsf) + 1j * numpy.random.rand(
    nwalkers, nk, nocc, nk, nbsf
)

GhalfaT = Ghalfa_batch.transpose(0, 3, 4, 1, 2)  # (nwalkers, nk, nbsf, nk, nocc)
GhalfbT = Ghalfb_batch.transpose(0, 3, 4, 1, 2)  # (nwalkers, nk, nbsf, nk, nocc)
GhalfaT = GhalfaT.transpose(1, 3, 0, 2, 4).copy()  # (nk, nk, nwalkers, nbsf, nocc)
GhalfbT = GhalfbT.transpose(1, 3, 0, 2, 4).copy()  # (nk, nk, nwalkers, nbsf, nocc)
GhalfaT2 = Ghalfa_batch.transpose(1, 3, 0, 2, 4).copy()  # (nk, nk, nwalkers, nocc, nbsf)
GhalfbT2 = Ghalfb_batch.transpose(1, 3, 0, 2, 4).copy()  # (nk, nk, nwalkers, nocc, nbsf)
rcholaT = rchola.transpose(3, 1, 0, 2, 4).copy()  # (nq, nk, naux, nocc, nbsf)
rcholbT = rcholb.transpose(3, 1, 0, 2, 4).copy()  # (nq, nk, naux, nocc, nbsf)
rcholbaraT = rcholbara.transpose(3, 1, 0, 2, 4).copy()  # (nq, nk, naux, nbsf, nocc)
rcholbarbT = rcholbarb.transpose(3, 1, 0, 2, 4).copy()  # (nq, nk, naux, nbsf, nocc)
kpq_mat = numpy.random.randint(0, nk, (nk, nk))

# Profile the function
lp = LineProfiler()
profiled_fn = lp(prof_kpt_symmchol_ecoul_kernel_uhf)
# profiled_fn = lp(kpt_symmchol_ecoul_kernel_uhf)
# ecoul = kpt_symmchol_ecoul_kernel_uhf(rcholaT, rcholbT, rcholbaraT, rcholbarbT, GhalfaT, GhalfbT, GhalfaT2, GhalfbT2, kpq_mat, Qset)
ecoul = kpt_symmchol_ecoul_kernel_uhf_opt(
    rcholaT, rcholbT, rcholbaraT, rcholbarbT, GhalfaT, GhalfbT, GhalfaT2, GhalfbT2, kpq_mat, Qset
)
profiled_fn(
    rcholaT, rcholbT, rcholbaraT, rcholbarbT, GhalfaT, GhalfbT, GhalfaT2, GhalfbT2, kpq_mat, Qset
)
lp.print_stats()
