from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from numba import jit
import numpy

def greens_function_kpt_single_det(walker_batch, trial, build_full=False):
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
    nup = trial.nalpha
    ndown = trial.nbeta
    nbsf = trial.nbasis
    nk = trial.nk

    phia = walker_batch.phia.reshape(walker_batch.nwalkers, nk, nbsf, nk, nup).copy()
    phib = walker_batch.phib.reshape(walker_batch.nwalkers, nk, nbsf, nk, ndown).copy()
    det = []
    signovlp = []
    logovlp = []
    for iw in range(walker_batch.nwalkers):
        ovlpt = numpy.zeros((nk, nup, nk, nup), dtype=numpy.complex128)
        for ik1 in range(nk):
            for ik2 in range(nk):
                ovlpt[ik1, :, ik2, :] = numpy.dot(phia[iw, ik2, :, ik1, :].T, trial.psi0a[ik2].conj())
        ovlpt = ovlpt.reshape(nk*nup, nk*nup)
        ovlpinvt = numpy.linalg.inv(ovlpt)
        walker_batch.Ghalfa[iw] = numpy.dot(ovlpinvt, walker_batch.phia[iw].T)
        Ghalfa_reshaped = walker_batch.Ghalfa[iw].reshape(nk, nup, nk, nbsf)
        Ga = numpy.zeros((nk, nbsf, nk, nbsf), dtype=numpy.complex128)
        if not trial.half_rotated or build_full:
            for ik1 in range(nk):
                for ik2 in range(nk):
                    Ga[ik1, :, ik2, :] = numpy.dot(trial.psi0a[ik1].conj(), Ghalfa_reshaped[iw, ik1, :, ik2, :])
            walker_batch.Ga[iw] = Ga.reshape(nk * nbsf, nk * nbsf)
        sign_a, log_ovlp_a = xp.linalg.slogdet(ovlpt)
        sign_b, log_ovlp_b = 1.0, 0.0
        if ndown > 0 and not walker_batch.rhf:
            ovlpt = numpy.zeros((nk, ndown, nk, ndown), dtype=numpy.complex128)
            for ik1 in range(nk):
                for ik2 in range(nk):
                    ovlpt[ik1, :, ik2, :] = numpy.dot(phib[iw, ik2, :, ik1, :].T, trial.psi0b[ik2].conj())
            ovlpt = ovlpt.reshape(nk*ndown, nk*ndown)
            sign_b, log_ovlp_b = xp.linalg.slogdet(ovlpt)
            ovlpinvt = numpy.linalg.inv(ovlpt)
            walker_batch.Ghalfb[iw] = numpy.dot(ovlpinvt, walker_batch.phib[iw].T)
            Ghalfb_reshaped = walker_batch.Ghalfb[iw].reshape(nk, ndown, nk, nbsf)
            Gb = numpy.zeros((nk, nbsf, nk, nbsf), dtype=numpy.complex128)
            if not trial.half_rotated or build_full:
                for ik1 in range(nk):
                    for ik2 in range(nk):
                        Gb[iw, ik1, :, ik2, :] = numpy.dot(trial.psi0b[ik1].conj(), Ghalfb_reshaped[iw, ik1, :, ik2, :])
                walker_batch.Gb[iw] = Gb.reshape(nk * nbsf, nk * nbsf)
            det += [sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walker_batch.log_shift[iw])]
            signovlp += [sign_a * sign_b]
            logovlp += [log_ovlp_a + log_ovlp_b - walker_batch.log_shift[iw]]
        elif ndown > 0 and walker_batch.rhf:
            det += [sign_a * sign_a * xp.exp(log_ovlp_a + log_ovlp_a - walker_batch.log_shift[iw])]
            signovlp += [sign_a * sign_a]
            logovlp += [log_ovlp_a + log_ovlp_a - walker_batch.log_shift[iw]]
        elif ndown == 0:
            det += [sign_a * xp.exp(log_ovlp_a - walker_batch.log_shift[iw])]
            signovlp += [sign_a]
            logovlp += [log_ovlp_a - walker_batch.log_shift[iw]]

    det = xp.array(det, dtype=xp.complex128)
    signovlp = xp.array(signovlp, dtype=xp.complex128)
    logovlp = xp.array(logovlp, dtype=xp.float64)

    synchronize()

    return det, signovlp, logovlp


def greens_function_kpt_single_det_batch(walker_batch, trial, build_full=False):
    """Compute walker's green's function using only batched operations.

    Parameters
    ----------
    walker_batch : object
        SingleDetWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    ot : float64 / complex128
        Overlap with trial.
    """
    nup = trial.nalpha
    ndown = trial.nbeta
    nbsf = trial.nbasis
    nk = trial.nk
    phia = walker_batch.phia.reshape(walker_batch.nwalkers, nk, nbsf, nk, nup)
    if ndown > 0:
        phib = walker_batch.phib.reshape(walker_batch.nwalkers, nk, nbsf, nk, ndown)
    else:
        phib = None

    ovlp_a = xp.einsum("wlpki, lpj->wkilj", phia, trial.psi0a.conj(), optimize=True)
    ovlp_a = ovlp_a.reshape(walker_batch.nwalkers, nk * nup, nk * nup)
    ovlp_inv_a = xp.linalg.inv(ovlp_a)
    sign_a, log_ovlp_a = xp.linalg.slogdet(ovlp_a)

    # walker_batch.Ghalfa = xp.einsum("wij,wmj->wim", ovlp_inv_a, walker_batch.phia, optimize=True)
    walker_batch.Ghalfa = xp.matmul(ovlp_inv_a, walker_batch.phia.transpose(0, 2, 1))
    if not trial.half_rotated or build_full:
        Ga = xp.einsum(
            "kpi,wkilq->wkplq", trial.psi0a.conj(), walker_batch.Ghalfa, optimize=True
        )
        walker_batch.Ga = Ga.reshape(walker_batch.nwalkers, nk, nbsf, nk, nbsf)

    if ndown > 0 and not walker_batch.rhf:
        ovlp_b = xp.einsum("wlpki, lpj->wkilj", phib, trial.psi0b.conj(), optimize=True)
        ovlp_b = ovlp_b.reshape(walker_batch.nwalkers, nk * ndown, nk * ndown)
        ovlp_inv_b = xp.linalg.inv(ovlp_b)
        sign_b, log_ovlp_b = xp.linalg.slogdet(ovlp_b)

        walker_batch.Ghalfb = xp.matmul(ovlp_inv_b, walker_batch.phib.transpose(0, 2, 1))
        if not trial.half_rotated or build_full:
            Gb = xp.einsum(
                "kpi,wkilq->wkplq", trial.psi0b.conj(), walker_batch.Ghalfb, optimize=True
            )
            walker_batch.Gb = Gb.reshape(walker_batch.nwalkers, nk, nbsf, nk, nbsf)
        ot = sign_a * sign_b * xp.exp(log_ovlp_a + log_ovlp_b - walker_batch.log_shift)
        sgnot = sign_a * sign_b
        logot = log_ovlp_a + log_ovlp_b - walker_batch.log_shift
    elif ndown > 0 and walker_batch.rhf:
        ot = sign_a * sign_a * xp.exp(log_ovlp_a + log_ovlp_a - walker_batch.log_shift)
        sgnot = sign_a * sign_a
        logot = log_ovlp_a + log_ovlp_a - walker_batch.log_shift
    elif ndown == 0:
        ot = sign_a * xp.exp(log_ovlp_a - walker_batch.log_shift)
        sgnot = sign_a
        logot = log_ovlp_a - walker_batch.log_shift

    synchronize()

    return ot, sgnot, logot
