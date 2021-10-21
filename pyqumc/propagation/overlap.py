import numpy
import scipy.linalg

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
        calc_overlap = calc_overlap_single_det
    elif trial.name == "MultiSlater" and trial.ndets > 1:
        calc_overlap = calc_overlap_multi_det
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
    na = walker_batch.nup
    nb = walker_batch.ndown
    ot = numpy.zeros(walker_batch.nwalkers, dtype=numpy.complex128)

    for iw in range(walker_batch.nwalkers):
        Oalpha = numpy.dot(trial.psi[:,:na].conj().T, walker_batch.phi[iw][:,:na])
        sign_a, logdet_a = numpy.linalg.slogdet(Oalpha)
        logdet_b, sign_b = 0.0, 1.0
        if nb > 0:
            Obeta = numpy.dot(trial.psi[:,na:].conj().T, walker_batch.phi[iw][:,na:])
            sign_b, logdet_b = numpy.linalg.slogdet(Obeta)

        ot[iw] = sign_a*sign_b*numpy.exp(logdet_a+logdet_b-walker_batch.log_shift[iw])
    
    return ot


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
            Oup = numpy.dot(det[:,:nup].conj().T, walker_batch.phi[iw,:,:nup])
            Odn = numpy.dot(det[:,nup:].conj().T, walker_batch.phi[iw,:,nup:])
            sign_a, logdet_a = numpy.linalg.slogdet(Oup)
            sign_b, logdet_b = numpy.linalg.slogdet(Odn)
            walker_batch.det_ovlpas[iw,i] = sign_a*numpy.exp(logdet_a)
            walker_batch.det_ovlpbs[iw,i] = sign_b*numpy.exp(logdet_b)
            walker_batch.det_weights[iw,i] = trial.coeffs[i].conj() * walker_batch.det_ovlpas[iw,i] * walker_batch.det_ovlpbs[iw,i]
    return numpy.einsum("wi->w", walker_batch.det_weights)

