import numpy
from numba import jit
import math
from line_profiler import LineProfiler

# @jit(nopython=True, fastmath=True)
# def construct_VHS_kernel_symm(chol, xshifted, nk, nbasis, nwalkers, ikpq_mat, Qset):
#     VHS = numpy.zeros((nwalkers, nk, nbasis, nk, nbasis), dtype=numpy.complex128)
#     for iq in range(len(Qset)):
#         iq_real = Qset[iq]
#         for ik in range(nk):
#             ikpq = ikpq_mat[iq_real, ik]
#             x_iq = .5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
#             xconj_iq = .5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
#             cholkq = chol[:, ik, :, iq, :].copy().reshape(-1, nbasis*nbasis)
#             cholkqT = chol[:, ik, :, iq, :].transpose(0, 2, 1).copy().reshape(-1, nbasis*nbasis)
#             for iw in range(nwalkers):
#                 # VHS[iw, ik, ikpq] += numpy.einsum('wx, xpr -> wpr', x_iq[iw], chol[:, ik, :, iq, :])
#                 VHS[iw, ik, ikpq] += x_iq[iw] @ cholkq
#                 VHS[iw, ikpq, ik] += xconj_iq[iw] @ cholkqT.conj()

#     VHS = VHS.reshape(nwalkers, nk, nk, nbasis, nbasis).transpose(0, 1, 3, 2, 4).copy()
#     VHS = VHS.reshape(nwalkers, nk * nbasis, nk * nbasis)
#     return VHS


@jit(nopython=True, fastmath=True)
def construct_VHS_kernel_symm(chol, xshifted, nk, nbasis, nwalkers, ikpq_mat, Qset):
    VHS = numpy.zeros((nwalkers, nk, nk, nbasis * nbasis), dtype=numpy.complex128)
    nchol = chol.shape[2]
    for iq in range(len(Qset)):
        iq_real = Qset[iq]
        for ik in range(nk):
            ikpq = ikpq_mat[iq_real, ik]
            x_iq = 0.5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
            xconj_iq = 0.5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
            cholkq = chol[iq, ik].reshape(-1, nbasis * nbasis)
            cholkqT = chol[iq, ik].transpose(0, 2, 1).copy().reshape(-1, nbasis * nbasis)
            # VHS[iw, ik, ikpq] += numpy.einsum('wx, xpr -> wpr', x_iq[iw], chol[:, ik, :, iq, :])
            for iw in range(nwalkers):
                VHS[iw, ik, ikpq] += x_iq[iw] @ cholkq
                VHS[iw, ikpq, ik] += xconj_iq[iw] @ cholkqT.conj()
            # for iw in range(nwalkers):
            #     for ig in range(nchol):
            #         VHS[iw, ik, ikpq] += x_iq[iw, ig] * cholkq[ig]
            #         VHS[iw, ikpq, ik] += xconj_iq[iw, ig] * cholkqT[ig].conj()

    VHS = VHS.reshape(nwalkers, nk, nk, nbasis, nbasis).transpose(0, 1, 3, 2, 4).copy()
    VHS = VHS.reshape(nwalkers, nk * nbasis, nk * nbasis)
    return VHS


@jit(nopython=True, fastmath=True)
def construct_VHS_kernel_joonho(chol, xshifted, nk, nbasis, nwalkers, ikpq_mat, Qset):
    VHS = numpy.zeros((nk, nk, nwalkers, nbasis * nbasis), dtype=numpy.complex128)
    nchol = chol.shape[2]
    for iq in range(len(Qset)):
        iq_real = Qset[iq]
        for ik in range(nk):
            ikpq = ikpq_mat[iq_real, ik]
            x_iq = 0.5 * (1j * xshifted[0, :, :, iq] + xshifted[1, :, :, iq])
            xconj_iq = 0.5 * (1j * xshifted[0, :, :, iq] - xshifted[1, :, :, iq])
            cholkq = chol[iq, ik].reshape(-1, nbasis * nbasis)
            # cholkqT = chol[iq, ik].transpose(0, 2, 1).copy().reshape(-1, nbasis*nbasis)
            # VHS[iw, ik, ikpq] += numpy.einsum('wx, xpr -> wpr', x_iq[iw], chol[:, ik, :, iq, :])
            # for iw in range(nwalkers):
            #     VHS[iw, ik, ikpq] += x_iq[iw] @ cholkq
            #     VHS[iw, ikpq, ik] += xconj_iq[iw] @ cholkqT.conj()
            VHS[ik, ikpq] += x_iq @ cholkq
            XL = xconj_iq @ cholkq.conj()
            XL = XL.reshape(nwalkers, nbasis, nbasis).transpose(0, 2, 1).copy()
            VHS[ikpq, ik] += XL.reshape(nwalkers, nbasis * nbasis)

    VHS = VHS.reshape(nk, nk, nwalkers, nbasis, nbasis).transpose(2, 0, 3, 1, 4).copy()
    VHS = VHS.reshape(nwalkers, nk * nbasis, nk * nbasis)
    return VHS


def prof_const_VHS_kernel(chol, xshifted, nk, nbasis, nwalkers, ikpq_mat, Qset):
    return construct_VHS_kernel_symm(chol, xshifted, nk, nbasis, nwalkers, ikpq_mat, Qset)


def prof_const_VHS_kernel_joonho(chol, xshifted, nk, nbasis, nwalkers, ikpq_mat, Qset):
    return construct_VHS_kernel_joonho(chol, xshifted, nk, nbasis, nwalkers, ikpq_mat, Qset)


nk = 27
nchol = 210
nbsf = 26
nwalkers = 10

nq = nk // 2 + 1
Qset = numpy.arange(nq)

kpq_mat = numpy.random.randint(0, nk, (nk, nk))
# chol = numpy.random.rand(nchol, nk, nbsf, nq, nbsf) + 1j * numpy.random.rand(nchol, nk, nbsf, nq, nbsf)
chol = numpy.random.rand(nq, nk, nchol, nbsf, nbsf) + 1j * numpy.random.rand(
    nq, nk, nchol, nbsf, nbsf
)
xshifted = numpy.random.rand(2, nwalkers, nchol, nq) + 1j * numpy.random.rand(
    2, nwalkers, nchol, nq
)

# Profile the function
# lp = LineProfiler()
# profiled_fn = lp(prof_const_VHS_kernel)
# profiled_fn_joonho = lp(prof_const_VHS_kernel_joonho)
VHS = construct_VHS_kernel_symm(chol, xshifted, nk, nbsf, nwalkers, kpq_mat, Qset)
VHS_joonho = construct_VHS_kernel_joonho(chol, xshifted, nk, nbsf, nwalkers, kpq_mat, Qset)
print(f"VHS difference: {numpy.linalg.norm(VHS.ravel() - VHS_joonho.ravel())}")
# profiled_fn(chol, xshifted, nk, nbsf, nwalkers, kpq_mat, Qset)
# profiled_fn_joonho(chol, xshifted, nk, nbsf, nwalkers, kpq_mat, Qset)
# lp.print_stats()
