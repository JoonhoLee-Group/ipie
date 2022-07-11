import numpy
import scipy.linalg

from ipie.utils.linalg import minor_mask
from ipie.utils.misc import is_cupy


# Later we will add walker kinds as an input too
def get_greens_function(trial):
    """Wrapper to select the calc_overlap function

    Parameters
    ----------
    trial : class
        Trial wavefunction object.

    Returns
    -------
    propagator : class or None
        Propagator object.
    """
    if trial.name == "MultiSlater" and trial.ndets == 1:
        compute_greens_function = _greens_function_single_det
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.wicks == False:
        compute_greens_function = greens_function_multi_det
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.wicks == True:
        # compute_greens_function = greens_function_multi_det
        compute_greens_function = greens_function_multi_det_wicks
    else:
        compute_greens_function = None

    return compute_greens_function


def greens_function(walker_batch, trial):
    compute_greens_function = get_greens_function(trial)
    return compute_greens_function(walker_batch, trial)


def greens_function_single_det(walker_batch, trial):
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
    if is_cupy(
        trial.psi
    ):  # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy

        assert cupy.is_available()
        array = cupy.array
        dot = cupy.dot
        exp = cupy.exp
        inv = cupy.linalg.inv
        slogdet = cupy.linalg.slogdet
    else:
        array = numpy.array
        dot = numpy.dot
        exp = numpy.exp
        inv = scipy.linalg.inv
        slogdet = numpy.linalg.slogdet

    nup = walker_batch.nup
    ndown = walker_batch.ndown

    det = []

    for iw in range(walker_batch.nwalkers):
        ovlp = dot(walker_batch.phia[iw].T, trial.psi[:, :nup].conj())
        ovlp_inv = inv(ovlp)
        walker_batch.Ghalfa[iw] = dot(ovlp_inv, walker_batch.phia[iw].T)
        walker_batch.Ga[iw] = dot(trial.psi[:, :nup].conj(), walker_batch.Ghalfa[iw])
        sign_a, log_ovlp_a = slogdet(ovlp)
        sign_b, log_ovlp_b = 1.0, 0.0
        if ndown > 0 and not walker_batch.rhf:
            ovlp = dot(walker_batch.phib[iw].T, trial.psi[:, nup:].conj())
            sign_b, log_ovlp_b = slogdet(ovlp)
            walker_batch.Ghalfb[iw] = dot(inv(ovlp), walker_batch.phib[iw].T)
            walker_batch.Gb[iw] = dot(
                trial.psi[:, nup:].conj(), walker_batch.Ghalfb[iw]
            )
            det += [
                sign_a
                * sign_b
                * exp(log_ovlp_a + log_ovlp_b - walker_batch.log_shift[iw])
            ]
        elif ndown > 0 and walker_batch.rhf:
            det += [
                sign_a
                * sign_a
                * exp(log_ovlp_a + log_ovlp_a - walker_batch.log_shift[iw])
            ]
        elif ndown == 0:
            det += [sign_a * exp(log_ovlp_a - walker_batch.log_shift)]

    det = array(det, dtype=numpy.complex128)

    return det


def _greens_function_single_det(walker_batch, trial):
    # Hack for poor cpu batched performance for the moment.
    # einsum performs poorly on cpu for these operations
    # so use for loop instead for the time being.
    # see: https://github.com/numpy/numpy/issues/19647
    if is_cupy(trial.psi):
        return greens_function_single_det_batch(walker_batch, trial)
    else:
        return greens_function_single_det(walker_batch, trial)


def greens_function_single_det_batch(walker_batch, trial):
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
    if is_cupy(
        trial.psi
    ):  # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy

        assert cupy.is_available()
        array = cupy.array
        dot = cupy.dot
        exp = cupy.exp
        einsum = cupy.einsum
        inv = cupy.linalg.inv
        slogdet = cupy.linalg.slogdet
    else:
        array = numpy.array
        dot = numpy.dot
        exp = numpy.exp
        einsum = numpy.einsum
        inv = numpy.linalg.inv
        slogdet = numpy.linalg.slogdet

    nup = walker_batch.nup
    ndown = walker_batch.ndown

    ovlp_a = einsum("wmi,mj->wij", walker_batch.phia, trial.psia.conj(), optimize=True)
    ovlp_inv_a = inv(ovlp_a)
    sign_a, log_ovlp_a = slogdet(ovlp_a)

    walker_batch.Ghalfa = einsum(
        "wij,wmj->wim", ovlp_inv_a, walker_batch.phia, optimize=True
    )
    walker_batch.Ga = einsum(
        "mi,win->wmn", trial.psia.conj(), walker_batch.Ghalfa, optimize=True
    )

    if ndown > 0 and not walker_batch.rhf:
        ovlp_b = einsum(
            "wmi,mj->wij", walker_batch.phib, trial.psib.conj(), optimize=True
        )
        ovlp_inv_b = inv(ovlp_b)

        sign_b, log_ovlp_b = slogdet(ovlp_b)
        walker_batch.Ghalfb = einsum(
            "wij,wmj->wim", ovlp_inv_b, walker_batch.phib, optimize=True
        )
        walker_batch.Gb = einsum(
            "mi,win->wmn", trial.psib.conj(), walker_batch.Ghalfb, optimize=True
        )
        ot = sign_a * sign_b * exp(log_ovlp_a + log_ovlp_b - walker_batch.log_shift)
    elif ndown > 0 and walker_batch.rhf:
        ot = sign_a * sign_a * exp(log_ovlp_a + log_ovlp_a - walker_batch.log_shift)
    elif ndown == 0:
        ot = sign_a * exp(log_ovlp_a - walker_batch.log_shift)

    return ot


def greens_function_multi_det(walker_batch, trial):
    """Compute walker's green's function.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    det : float64 / complex128
        Determinant of overlap matrix.
    """
    nup = walker_batch.nup
    walker_batch.Ga.fill(0.0)
    walker_batch.Gb.fill(0.0)
    tot_ovlps = numpy.zeros(walker_batch.nwalkers, dtype=numpy.complex128)
    for iw in range(walker_batch.nwalkers):
        for (ix, detix) in enumerate(trial.psi):
            # construct "local" green's functions for each component of psi_T
            Oup = numpy.dot(walker_batch.phia[iw].T, detix[:, :nup].conj())
            # det(A) = det(A^T)
            sign_a, logdet_a = numpy.linalg.slogdet(Oup)
            walker_batch.det_ovlpas[iw, ix] = sign_a * numpy.exp(logdet_a)
            if abs(walker_batch.det_ovlpas[iw, ix]) < 1e-16:
                continue

            Odn = numpy.dot(walker_batch.phib[iw].T, detix[:, nup:].conj())
            sign_b, logdet_b = numpy.linalg.slogdet(Odn)
            walker_batch.det_ovlpbs[iw, ix] = sign_b * numpy.exp(logdet_b)
            ovlp = walker_batch.det_ovlpas[iw, ix] * walker_batch.det_ovlpbs[iw, ix]
            if abs(ovlp) < 1e-16:
                continue

            inv_ovlp = scipy.linalg.inv(Oup)
            walker_batch.Gihalfa[iw, ix, :, :] = numpy.dot(
                inv_ovlp, walker_batch.phia[iw].T
            )
            walker_batch.Gia[iw, ix, :, :] = numpy.dot(
                detix[:, :nup].conj(), walker_batch.Gihalfa[iw, ix, :, :]
            )

            inv_ovlp = scipy.linalg.inv(Odn)
            walker_batch.Gihalfb[iw, ix, :, :] = numpy.dot(
                inv_ovlp, walker_batch.phib[iw].T
            )
            walker_batch.Gib[iw, ix, :, :] = numpy.dot(
                detix[:, nup:].conj(), walker_batch.Gihalfb[iw, ix, :, :]
            )

            tot_ovlps[iw] += trial.coeffs[ix].conj() * ovlp
            walker_batch.det_weights[iw, ix] = trial.coeffs[ix].conj() * ovlp

            walker_batch.Ga[iw] += (
                walker_batch.Gia[iw, ix, :, :] * ovlp * trial.coeffs[ix].conj()
            )
            walker_batch.Gb[iw] += (
                walker_batch.Gib[iw, ix, :, :] * ovlp * trial.coeffs[ix].conj()
            )

        walker_batch.Ga[iw] /= tot_ovlps[iw]
        walker_batch.Gb[iw] /= tot_ovlps[iw]

    return tot_ovlps


def greens_function_multi_det_wicks(walker_batch, trial):
    """Compute walker's green's function using Wick's theorem.

    Parameters
    ----------
    walker_batch : object
        MultiDetTrialWalkerBatch object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    det : float64 / complex128
        Determinant of overlap matrix.
    """
    tot_ovlps = numpy.zeros(walker_batch.nwalkers, dtype=numpy.complex128)
    nbasis = walker_batch.Ga.shape[-1]

    nup = walker_batch.nup
    ndown = walker_batch.ndown

    walker_batch.Ga.fill(0.0 + 0.0j)
    walker_batch.Gb.fill(0.0 + 0.0j)

    for iw in range(walker_batch.nwalkers):
        phia = walker_batch.phia[iw]  # walker wfn
        phib = walker_batch.phib[iw]  # walker wfn

        Oalpha = numpy.dot(trial.psi0a.conj().T, phia)
        sign_a, logdet_a = numpy.linalg.slogdet(Oalpha)
        logdet_b, sign_b = 0.0, 1.0
        Obeta = numpy.dot(trial.psi0b.conj().T, phib)
        sign_b, logdet_b = numpy.linalg.slogdet(Obeta)

        ovlp0 = sign_a * sign_b * numpy.exp(logdet_a + logdet_b)
        walker_batch.det_ovlpas[iw, 0] = sign_a * numpy.exp(logdet_a)
        walker_batch.det_ovlpbs[iw, 0] = sign_b * numpy.exp(logdet_b)
        ovlpa0 = walker_batch.det_ovlpas[iw, 0]
        ovlpab = walker_batch.det_ovlpbs[iw, 0]

        # G0, G0H = gab_spin(trial.psi0, phi, nup, ndown)
        G0a, G0Ha = gab_mod(trial.psi0a, phia)
        G0b, G0Hb = gab_mod(trial.psi0b, phib)
        walker_batch.G0a[iw] = G0a
        walker_batch.G0b[iw] = G0b
        walker_batch.Q0a[iw] = numpy.eye(nbasis) - walker_batch.G0a[iw]
        walker_batch.Q0b[iw] = numpy.eye(nbasis) - walker_batch.G0b[iw]

        G0a = walker_batch.G0a[iw]
        G0b = walker_batch.G0b[iw]
        Q0a = walker_batch.Q0a[iw]
        Q0b = walker_batch.Q0b[iw]

        ovlp = 0.0 + 0.0j
        ovlp += trial.coeffs[0].conj()

        walker_batch.Ga[iw] += G0a * trial.coeffs[0].conj()
        walker_batch.Gb[iw] += G0b * trial.coeffs[0].conj()

        walker_batch.CIa[iw].fill(0.0 + 0.0j)
        walker_batch.CIb[iw].fill(0.0 + 0.0j)

        for jdet in range(1, trial.ndets):
            nex_a = len(trial.cre_a[jdet])
            nex_b = len(trial.cre_b[jdet])

            det_a = numpy.zeros((nex_a, nex_a), dtype=numpy.complex128)
            det_b = numpy.zeros((nex_b, nex_b), dtype=numpy.complex128)

            for iex in range(nex_a):
                det_a[iex, iex] = G0a[trial.cre_a[jdet][iex], trial.anh_a[jdet][iex]]
                for jex in range(iex + 1, nex_a):
                    det_a[iex, jex] = G0a[
                        trial.cre_a[jdet][iex], trial.anh_a[jdet][jex]
                    ]
                    det_a[jex, iex] = G0a[
                        trial.cre_a[jdet][jex], trial.anh_a[jdet][iex]
                    ]
            for iex in range(nex_b):
                det_b[iex, iex] = G0b[trial.cre_b[jdet][iex], trial.anh_b[jdet][iex]]
                for jex in range(iex + 1, nex_b):
                    det_b[iex, jex] = G0b[
                        trial.cre_b[jdet][iex], trial.anh_b[jdet][jex]
                    ]
                    det_b[jex, iex] = G0b[
                        trial.cre_b[jdet][jex], trial.anh_b[jdet][iex]
                    ]

            walker_batch.det_ovlpas[iw, jdet] = (
                numpy.linalg.det(det_a) * trial.phase_a[jdet]
            )
            walker_batch.det_ovlpbs[iw, jdet] = (
                numpy.linalg.det(det_b) * trial.phase_b[jdet]
            )
            ovlpa = walker_batch.det_ovlpas[iw, jdet]
            ovlpb = walker_batch.det_ovlpbs[iw, jdet]

            ovlp += trial.coeffs[jdet].conj() * ovlpa * ovlpb

            # contribution 1 (disconnected diagrams)
            walker_batch.Ga[iw] += trial.coeffs[jdet].conj() * G0a * ovlpa * ovlpb
            walker_batch.Gb[iw] += trial.coeffs[jdet].conj() * G0b * ovlpa * ovlpb
            # intermediates for contribution 2 (connected diagrams)
            if nex_a == 1:
                walker_batch.CIa[iw, trial.anh_a[jdet][0], trial.cre_a[jdet][0]] += (
                    trial.coeffs[jdet].conj() * trial.phase_a[jdet] * ovlpb
                )
            elif nex_a == 2:
                p = trial.cre_a[jdet][0]
                q = trial.anh_a[jdet][0]
                r = trial.cre_a[jdet][1]
                s = trial.anh_a[jdet][1]
                walker_batch.CIa[iw, q, p] += (
                    trial.coeffs[jdet].conj() * trial.phase_a[jdet] * ovlpb * G0a[r, s]
                )
                walker_batch.CIa[iw, s, r] += (
                    trial.coeffs[jdet].conj() * trial.phase_a[jdet] * ovlpb * G0a[p, q]
                )
                walker_batch.CIa[iw, q, r] -= (
                    trial.coeffs[jdet].conj() * trial.phase_a[jdet] * ovlpb * G0a[p, s]
                )
                walker_batch.CIa[iw, s, p] -= (
                    trial.coeffs[jdet].conj() * trial.phase_a[jdet] * ovlpb * G0a[r, q]
                )
            elif nex_a == 3:
                p = trial.cre_a[jdet][0]
                q = trial.anh_a[jdet][0]
                r = trial.cre_a[jdet][1]
                s = trial.anh_a[jdet][1]
                t = trial.cre_a[jdet][2]
                u = trial.anh_a[jdet][2]

                walker_batch.CIa[iw, q, p] += (
                    trial.coeffs[jdet].conj()
                    * trial.phase_a[jdet]
                    * ovlpb
                    * (G0a[r, s] * G0a[t, u] - G0a[r, u] * G0a[t, s])
                )  # 0 0
                walker_batch.CIa[iw, s, p] -= (
                    trial.coeffs[jdet].conj()
                    * trial.phase_a[jdet]
                    * ovlpb
                    * (G0a[r, q] * G0a[t, u] - G0a[r, u] * G0a[t, q])
                )  # 0 1
                walker_batch.CIa[iw, u, p] += (
                    trial.coeffs[jdet].conj()
                    * trial.phase_a[jdet]
                    * ovlpb
                    * (G0a[r, q] * G0a[t, s] - G0a[r, s] * G0a[t, q])
                )  # 0 2

                walker_batch.CIa[iw, q, r] -= (
                    trial.coeffs[jdet].conj()
                    * trial.phase_a[jdet]
                    * ovlpb
                    * (G0a[p, s] * G0a[t, u] - G0a[p, u] * G0a[t, s])
                )  # 1 0
                walker_batch.CIa[iw, s, r] += (
                    trial.coeffs[jdet].conj()
                    * trial.phase_a[jdet]
                    * ovlpb
                    * (G0a[p, q] * G0a[t, u] - G0a[p, u] * G0a[t, q])
                )  # 1 1
                walker_batch.CIa[iw, u, r] -= (
                    trial.coeffs[jdet].conj()
                    * trial.phase_a[jdet]
                    * ovlpb
                    * (G0a[p, q] * G0a[t, s] - G0a[p, s] * G0a[t, q])
                )  # 1 2

                walker_batch.CIa[iw, q, t] += (
                    trial.coeffs[jdet].conj()
                    * trial.phase_a[jdet]
                    * ovlpb
                    * (G0a[p, s] * G0a[r, u] - G0a[p, u] * G0a[r, s])
                )  # 2 0
                walker_batch.CIa[iw, s, t] -= (
                    trial.coeffs[jdet].conj()
                    * trial.phase_a[jdet]
                    * ovlpb
                    * (G0a[p, q] * G0a[r, u] - G0a[p, u] * G0a[r, q])
                )  # 2 1
                walker_batch.CIa[iw, u, t] += (
                    trial.coeffs[jdet].conj()
                    * trial.phase_a[jdet]
                    * ovlpb
                    * (G0a[p, q] * G0a[r, s] - G0a[p, s] * G0a[r, q])
                )  # 2 2

            elif nex_a > 3:
                cofactor = numpy.zeros((nex_a - 1, nex_a - 1), dtype=numpy.complex128)
                for iex in range(nex_a):
                    p = trial.cre_a[jdet][iex]
                    for jex in range(nex_a):
                        q = trial.anh_a[jdet][jex]
                        cofactor[:, :] = minor_mask(det_b, iex, jex)
                        walker_batch.CIa[iw, q, p] += (
                            trial.coeffs[jdet].conj()
                            * trial.phase_a[jdet]
                            * ovlpb
                            * (-1) ** (iex + jex)
                            * numpy.linalg.det(cofactor)
                        )

            if nex_b == 1:
                walker_batch.CIb[iw, trial.anh_b[jdet][0], trial.cre_b[jdet][0]] += (
                    trial.coeffs[jdet].conj() * trial.phase_b[jdet] * ovlpa
                )
            elif nex_b == 2:
                p = trial.cre_b[jdet][0]
                q = trial.anh_b[jdet][0]
                r = trial.cre_b[jdet][1]
                s = trial.anh_b[jdet][1]
                walker_batch.CIb[iw, q, p] += (
                    trial.coeffs[jdet].conj() * trial.phase_b[jdet] * ovlpa * G0b[r, s]
                )
                walker_batch.CIb[iw, s, r] += (
                    trial.coeffs[jdet].conj() * trial.phase_b[jdet] * ovlpa * G0b[p, q]
                )
                walker_batch.CIb[iw, q, r] -= (
                    trial.coeffs[jdet].conj() * trial.phase_b[jdet] * ovlpa * G0b[p, s]
                )
                walker_batch.CIb[iw, s, p] -= (
                    trial.coeffs[jdet].conj() * trial.phase_b[jdet] * ovlpa * G0b[r, q]
                )
            elif nex_b == 3:
                p = trial.cre_b[jdet][0]
                q = trial.anh_b[jdet][0]
                r = trial.cre_b[jdet][1]
                s = trial.anh_b[jdet][1]
                t = trial.cre_b[jdet][2]
                u = trial.anh_b[jdet][2]

                walker_batch.CIb[iw, q, p] += (
                    trial.coeffs[jdet].conj()
                    * trial.phase_b[jdet]
                    * ovlpa
                    * (G0b[r, s] * G0b[t, u] - G0b[r, u] * G0b[t, s])
                )  # 0 0
                walker_batch.CIb[iw, s, p] -= (
                    trial.coeffs[jdet].conj()
                    * trial.phase_b[jdet]
                    * ovlpa
                    * (G0b[r, q] * G0b[t, u] - G0b[r, u] * G0b[t, q])
                )  # 0 1
                walker_batch.CIb[iw, u, p] += (
                    trial.coeffs[jdet].conj()
                    * trial.phase_b[jdet]
                    * ovlpa
                    * (G0b[r, q] * G0b[t, s] - G0b[r, s] * G0b[t, q])
                )  # 0 2

                walker_batch.CIb[iw, q, r] -= (
                    trial.coeffs[jdet].conj()
                    * trial.phase_b[jdet]
                    * ovlpa
                    * (G0b[p, s] * G0b[t, u] - G0b[p, u] * G0b[t, s])
                )  # 1 0
                walker_batch.CIb[iw, s, r] += (
                    trial.coeffs[jdet].conj()
                    * trial.phase_b[jdet]
                    * ovlpa
                    * (G0b[p, q] * G0b[t, u] - G0b[p, u] * G0b[t, q])
                )  # 1 1
                walker_batch.CIb[iw, u, r] -= (
                    trial.coeffs[jdet].conj()
                    * trial.phase_b[jdet]
                    * ovlpa
                    * (G0b[p, q] * G0b[t, s] - G0b[p, s] * G0b[t, q])
                )  # 1 2

                walker_batch.CIb[iw, q, t] += (
                    trial.coeffs[jdet].conj()
                    * trial.phase_b[jdet]
                    * ovlpa
                    * (G0b[p, s] * G0b[r, u] - G0b[p, u] * G0b[r, s])
                )  # 2 0
                walker_batch.CIb[iw, s, t] -= (
                    trial.coeffs[jdet].conj()
                    * trial.phase_b[jdet]
                    * ovlpa
                    * (G0b[p, q] * G0b[r, u] - G0b[p, u] * G0b[r, q])
                )  # 2 1
                walker_batch.CIb[iw, u, t] += (
                    trial.coeffs[jdet].conj()
                    * trial.phase_b[jdet]
                    * ovlpa
                    * (G0b[p, q] * G0b[r, s] - G0b[p, s] * G0b[r, q])
                )  # 2 2

            elif nex_b > 3:
                cofactor = numpy.zeros((nex_b - 1, nex_b - 1), dtype=numpy.complex128)
                for iex in range(nex_b):
                    p = trial.cre_b[jdet][iex]
                    for jex in range(nex_b):
                        q = trial.anh_b[jdet][jex]
                        cofactor[:, :] = minor_mask(det_b, iex, jex)
                        walker_batch.CIb[iw, q, p] += (
                            trial.coeffs[jdet].conj()
                            * trial.phase_b[jdet]
                            * ovlpa
                            * (-1) ** (iex + jex)
                            * numpy.linalg.det(cofactor)
                        )

        # contribution 2 (connected diagrams)
        walker_batch.Ga[iw] += Q0a.dot(walker_batch.CIa[iw]).dot(G0a)
        walker_batch.Gb[iw] += Q0b.dot(walker_batch.CIb[iw]).dot(G0b)

        # multiplying everything by reference overlap
        ovlp *= ovlp0
        walker_batch.Ga[iw] *= ovlp0
        walker_batch.Gb[iw] *= ovlp0

        walker_batch.Ga[iw] /= ovlp
        walker_batch.Gb[iw] /= ovlp

        tot_ovlps[iw] = ovlp

    return tot_ovlps


# Green's functions
def gab(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    # Todo: check energy evaluation at later point, i.e., if this needs to be
    # transposed. Shouldn't matter for Hubbard model.
    inv_O = scipy.linalg.inv((A.conj().T).dot(B))
    GAB = B.dot(inv_O.dot(A.conj().T))
    return GAB


def gab_mod(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    O = numpy.dot(B.T, A.conj())
    GHalf = numpy.dot(scipy.linalg.inv(O), B.T)
    G = numpy.dot(A.conj(), GHalf)
    return (G, GHalf)


def gab_spin(A, B, na, nb):
    GA, GAH = gab_mod(A[:, :na], B[:, :na])
    if nb > 0:
        GB, GBH = gab_mod(A[:, na:], B[:, na:])
    return numpy.array([GA, GB]), [GAH, GBH]


def gab_mod_ovlp(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    inv_O = scipy.linalg.inv(numpy.dot(B.T, A.conj()))
    GHalf = numpy.dot(inv_O, B.T)
    G = numpy.dot(A.conj(), GHalf)
    return (G, GHalf, inv_O)


def gab_multi_det(A, B, coeffs):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle = [B(A^{*T}B)^{-1}A^{*T}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent a multi-determinant trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.
    coeffs: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients. Assumed to be complex
        conjugated.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    # Todo: check energy evaluation at later point, i.e., if this needs to be
    # transposed. Shouldn't matter for Hubbard model.
    Gi = numpy.zeros(A.shape)
    overlaps = numpy.zeros(A.shape[1])
    for (ix, Aix) in enumerate(A):
        # construct "local" green's functions for each component of A
        # Todo: list comprehension here.
        inv_O = scipy.linalg.inv((Aix.conj().T).dot(B))
        Gi[ix] = (B.dot(inv_O.dot(Aix.conj().T))).T
        overlaps[ix] = 1.0 / scipy.linalg.det(inv_O)
    denom = numpy.dot(coeffs, overlaps)
    return numpy.einsum("i,ijk,i->jk", coeffs, Gi, overlaps) / denom


def gab_multi_ghf_full(A, B, coeffs, bp_weights):
    """Green's function for back propagation.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.
    coeffs: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients. Assumed to be complex
        conjugated.
    bp_weights : :class:`numpy.ndarray`
        Factors arising from GS orthogonalisation.

    Returns
    -------
    G : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    M = A.shape[1] // 2
    Gi, overlaps = construct_multi_ghf_gab(A, B, coeffs)
    scale = max(max(bp_weights), max(overlaps))
    full_weights = bp_weights * coeffs * overlaps / scale
    denom = sum(full_weights)
    G = numpy.einsum("i,ijk->jk", full_weights, Gi) / denom

    return G


def gab_multi_ghf(A, B, coeffs, Gi=None, overlaps=None):
    """Construct components of multi-ghf trial wavefunction.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.
    Gi : :class:`numpy.ndarray`
        Array to store components of G. Default: None.
    overlaps : :class:`numpy.ndarray`
        Array to overlaps. Default: None.

    Returns
    -------
    Gi : :class:`numpy.ndarray`
        Array to store components of G. Default: None.
    overlaps : :class:`numpy.ndarray`
        Array to overlaps. Default: None.
    """
    M = B.shape[0] // 2
    if Gi is None:
        Gi = numpy.zeros(shape=(A.shape[0], A.shape[1], A.shape[1]), dtype=A.dtype)
    if overlaps is None:
        overlaps = numpy.zeros(A.shape[0], dtype=A.dtype)
    for (ix, Aix) in enumerate(A):
        # construct "local" green's functions for each component of A
        # Todo: list comprehension here.
        inv_O = scipy.linalg.inv((Aix.conj().T).dot(B))
        Gi[ix] = B.dot(inv_O.dot(Aix.conj().T))
        overlaps[ix] = 1.0 / scipy.linalg.det(inv_O)
    return (Gi, overlaps)


def gab_multi_det_full(A, B, coeffsA, coeffsB, GAB, weights):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle = [B(A^{*T}B)^{-1}A^{*T}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    .. todo: Fix docstring

    Here we assume both A and B are multi-determinant expansions.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Array containing elements of multi-determinant matrix representation of
        the ket used to construct G.
    coeffsA: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients for wavefunction A. Assumed to
        be complex conjugated.
    coeffsB: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients for wavefunction A. Assumed to
        be complex conjugated.
    GAB : :class:`numpy.ndarray`
        Matrix of Green's functions.
    weights : :class:`numpy.ndarray`
        Matrix of weights needed to construct G

    Returns
    -------
    G : :class:`numpy.ndarray`
        Full Green's function.
    """
    for ix, (Aix, cix) in enumerate(zip(A, coeffsA)):
        for iy, (Biy, ciy) in enumerate(zip(B, coeffsB)):
            # construct "local" green's functions for each component of A
            inv_O = scipy.linalg.inv((Aix.conj().T).dot(Biy))
            GAB[ix, iy] = (Biy.dot(inv_O)).dot(Aix.conj().T)
            GAB[ix, iy] = (Biy.dot(inv_O)).dot(Aix.conj().T)
            weights[ix, iy] = cix * (ciy.conj()) / scipy.linalg.det(inv_O)
    denom = numpy.sum(weights)
    G = numpy.einsum("ij,ijkl->kl", weights, GAB) / denom
    return G
