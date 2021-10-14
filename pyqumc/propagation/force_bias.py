import numpy

def construct_force_bias_batch(hamiltonian, walker_batch, trial):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    Ghalf : :class:`numpy.ndarray`
        Half-rotated walker's Green's function.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    Ghalfa = walker_batch.Ghalfa.reshape(walker_batch.nwalkers, walker_batch.nup*hamiltonian.nbasis)
    Ghalfb = walker_batch.Ghalfb.reshape(walker_batch.nwalkers, walker_batch.ndown*hamiltonian.nbasis)
    if numpy.isrealobj(trial._rchola) and numpy.isrealobj(trial._rcholb):
        vbias_batch_real = trial._rchola.dot(Ghalfa.T.real) + trial._rcholb.dot(Ghalfb.T.real)
        vbias_batch_imag = trial._rchola.dot(Ghalfa.T.imag) + trial._rcholb.dot(Ghalfb.T.imag)
        vbias_batch = numpy.empty((walker_batch.nwalkers, hamiltonian.nchol), dtype=numpy.complex128)
        vbias_batch.real = vbias_batch_real.T.copy()
        vbias_batch.imag = vbias_batch_imag.T.copy()
        return vbias_batch
    else:    
        vbias_batch_tmp = trial._rchola.dot(Ghalfa.T) + trial._rcholb.dot(Ghalfb.T)
        return vbias_batch_tmp.T
