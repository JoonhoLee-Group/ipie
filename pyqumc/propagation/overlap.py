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
    else:
        calc_overlap = None

    return calc_overlap

def calc_overlap_single_det(walker_batch, trial):
    """Caculate overlap with trial wavefunction.

    Parameters
    ----------
    trial : object
        Trial wavefunction object.

    Returns
    -------
    ot : float / complex
        Overlap.
    """
    na = walker_batch.ndown
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
