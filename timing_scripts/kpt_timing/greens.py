import numpy
from numba import jit
from line_profiler import LineProfiler


def greens_function_kpt_single_det(phia, psi0a, nwalkers, nbsf, nk, nup):
    """Compute walker's green's function.

    Parameters
    ----------
    walker_batch : object
        SingleDetWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    det : float64 / complex128
        Determinant of overlap matrix.
    """

    phia_res = phia.reshape(nwalkers, nk, nbsf, nk, nup).copy()
    Ghalfa = numpy.zeros((nwalkers, nk * nup, nk * nbsf), dtype=numpy.complex128)
    det = []
    for iw in range(nwalkers):
        ovlpt = numpy.zeros((nk, nup, nk, nup), dtype=numpy.complex128)
        for ik1 in range(nk):
            for ik2 in range(nk):
                ovlpt[ik1, :, ik2, :] = numpy.dot(phia_res[iw, ik2, :, ik1, :].T, psi0a[ik2].conj())
        ovlpt = ovlpt.reshape(nk * nup, nk * nup)
        ovlpinvt = numpy.linalg.inv(ovlpt)
        Ghalfa[iw] = numpy.dot(ovlpinvt, phia[iw].T)
        Ghalfa_reshaped = Ghalfa[iw].reshape(nk, nup, nk, nbsf)
        sign_a, log_ovlp_a = numpy.linalg.slogdet(ovlpt)

        det += [sign_a * numpy.exp(log_ovlp_a)]

    det = numpy.array(det, dtype=numpy.complex128)

    return det


nwalkers = 10
nbsf = 26
nk = 27
nup, ndown = 4, 4
psi0a = numpy.random.rand(nk, nbsf, nup)
phia = numpy.random.rand(nwalkers, nk * nbsf, nk * nup)

greens_function_kpt_single_det(phia, psi0a, nwalkers, nbsf, nk, nup)
lp = LineProfiler()

profiled_fn = lp(greens_function_kpt_single_det)
profiled_fn(phia, psi0a, nwalkers, nbsf, nk, nup)
lp.print_stats()
