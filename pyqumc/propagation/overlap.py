import numpy
import scipy.linalg
from pyqumc.estimators.greens_function import gab_spin, gab_mod
from pyqumc.utils.misc import is_cupy

# Later we will add walker kinds as an input too
def get_calc_overlap(trial):
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
        # calc_overlap = calc_overlap_single_det
        calc_overlap = calc_overlap_single_det_batch
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.wicks == False:
        calc_overlap = calc_overlap_multi_det
    elif trial.name == "MultiSlater" and trial.ndets > 1 and trial.wicks == True:
        # calc_overlap = calc_overlap_multi_det
        calc_overlap = calc_overlap_multi_det_wicks
    else:
        calc_overlap = None

    return calc_overlap

def calc_overlap_single_det(walker_batch, trial):
    """Caculate overlap with single det trial wavefunction.

    Parameters
    ----------
    walker_batch : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    ot : float / complex
        Overlap.
    """
    if (is_cupy(trial.psi)):
        import cupy
        assert(cupy.is_available())
        zeros = cupy.zeros
        dot = cupy.dot
        slogdet = cupy.linalg.slogdet
        exp = cupy.exp
    else:
        zeros = numpy.zeros
        dot = numpy.dot
        slogdet = numpy.linalg.slogdet
        exp = numpy.exp

    na = walker_batch.nup
    nb = walker_batch.ndown
    ot = zeros(walker_batch.nwalkers, dtype=numpy.complex128)

    for iw in range(walker_batch.nwalkers):
        Oalpha = dot(trial.psia.conj().T, walker_batch.phia[iw])
        sign_a, logdet_a = slogdet(Oalpha)
        logdet_b, sign_b = 0.0, 1.0
        if nb > 0:
            Obeta = dot(trial.psib.conj().T, walker_batch.phib[iw])
            sign_b, logdet_b = slogdet(Obeta)

        ot[iw] = sign_a*sign_b*exp(logdet_a+logdet_b-walker_batch.log_shift[iw])
    
    return ot

def calc_overlap_single_det_batch(walker_batch, trial):
    """Caculate overlap with single det trial wavefunction.

    Parameters
    ----------
    walker_batch : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    ot : float / complex
        Overlap.
    """
    if (is_cupy(trial.psi)):
        import cupy
        assert(cupy.is_available())
        zeros = cupy.zeros
        dot = cupy.dot
        einsum = cupy.einsum
        slogdet = cupy.linalg.slogdet
        exp = cupy.exp
    else:
        zeros = numpy.zeros
        dot = numpy.dot
        einsum = numpy.einsum
        slogdet = numpy.linalg.slogdet
        exp = numpy.exp

    nup = walker_batch.nup
    ndown = walker_batch.ndown
    ovlp_a = einsum("wmi,mj->wij", walker_batch.phia, trial.psia.conj(), optimize = True)
    sign_a, log_ovlp_a = slogdet(ovlp_a)

    if ndown > 0 and not walker_batch.rhf:
        ovlp_b = einsum("wmi,mj->wij", walker_batch.phib, trial.psib.conj(), optimize = True)
        sign_b, log_ovlp_b = slogdet(ovlp_b)
        ot = sign_a*sign_b*exp(log_ovlp_a+log_ovlp_b-walker_batch.log_shift)
    elif ndown > 0 and walker_batch.rhf:
        ot = sign_a*sign_a*exp(log_ovlp_a+log_ovlp_a-walker_batch.log_shift)
    elif ndown == 0:
        ot = sign_a*exp(log_ovlp_a-walker_batch.log_shift)

    return ot

def calc_overlap_multi_det_wicks(walker_batch, trial):
    """Calculate overlap with multidet trial wavefunction using Wick's Theorem.

    Parameters
    ----------
    walker_batch : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    ovlps : float / complex
        Overlap.
    """
    psi0a = trial.psi0a # reference det
    psi0b = trial.psi0b # reference det

    na = walker_batch.nup
    nb = walker_batch.ndown

    ovlps = []
    for iw in range(walker_batch.nwalkers):
        phia = walker_batch.phia[iw]
        Oalpha = numpy.dot(psi0a.conj().T, phia)
        sign_a, logdet_a = numpy.linalg.slogdet(Oalpha)
        logdet_b, sign_b = 0.0, 1.0
        
        phib = walker_batch.phib[iw]
        Obeta = numpy.dot(psi0b.conj().T, phib)
        sign_b, logdet_b = numpy.linalg.slogdet(Obeta)

        ovlp0 = sign_a*sign_b*numpy.exp(logdet_a+logdet_b)

        G0a, G0Ha = gab_mod(psi0a, phia)
        G0b, G0Hb = gab_mod(psi0b, phib)
        # G0, G0H = gab_spin(psi0, phi, na, nb)
        # G0a = G0[0]
        # G0b = G0[1]

        ovlp = 0.0 + 0.0j
        ovlp += trial.coeffs[0].conj()
        for jdet in range(1, trial.ndets):
            nex_a = len(trial.anh_a[jdet])
            nex_b = len(trial.anh_b[jdet])

            det_a = numpy.zeros((nex_a,nex_a), dtype=numpy.complex128)    
            det_b = numpy.zeros((nex_b,nex_b), dtype=numpy.complex128)    

            for iex in range(nex_a):
                det_a[iex,iex] = G0a[trial.cre_a[jdet][iex],trial.anh_a[jdet][iex]]
                for jex in range(iex+1, nex_a):
                    det_a[iex, jex] = G0a[trial.cre_a[jdet][iex],trial.anh_a[jdet][jex]]
                    det_a[jex, iex] = G0a[trial.cre_a[jdet][jex],trial.anh_a[jdet][iex]]

            for iex in range(nex_b):
                det_b[iex,iex] = G0b[trial.cre_b[jdet][iex],trial.anh_b[jdet][iex]]
                for jex in range(iex+1, nex_b):
                    det_b[iex, jex] = G0b[trial.cre_b[jdet][iex],trial.anh_b[jdet][jex]]
                    det_b[jex, iex] = G0b[trial.cre_b[jdet][jex],trial.anh_b[jdet][iex]]
            
            tmp = trial.coeffs[jdet].conj() * numpy.linalg.det(det_a) * numpy.linalg.det(det_b) * trial.phase_a[jdet] * trial.phase_b[jdet]
            ovlp += tmp
        ovlp *= ovlp0

        ovlps += [ovlp]

    ovlps = numpy.array(ovlps, dtype = numpy.complex128)

    return ovlps


def calc_overlap_multi_det(walker_batch, trial):
    """Caculate overlap with multidet trial wavefunction.

    Parameters
    ----------
    walker_batch : object
        WalkerBatch object (this stores some intermediates for the particular trial wfn).
    trial : object
        Trial wavefunction object.

    Returns
    -------
    ovlp : float / complex
        Overlap.
    """
    nup = walker_batch.nup
    for iw in range(walker_batch.nwalkers):
        for (i, det) in enumerate(trial.psi):
            Oup = numpy.dot(det[:,:nup].conj().T, walker_batch.phia[iw])
            Odn = numpy.dot(det[:,nup:].conj().T, walker_batch.phib[iw])
            sign_a, logdet_a = numpy.linalg.slogdet(Oup)
            sign_b, logdet_b = numpy.linalg.slogdet(Odn)
            walker_batch.det_ovlpas[iw,i] = sign_a*numpy.exp(logdet_a)
            walker_batch.det_ovlpbs[iw,i] = sign_b*numpy.exp(logdet_b)
            walker_batch.det_weights[iw,i] = trial.coeffs[i].conj() * walker_batch.det_ovlpas[iw,i] * walker_batch.det_ovlpbs[iw,i]
    return numpy.einsum("wi->w", walker_batch.det_weights)

