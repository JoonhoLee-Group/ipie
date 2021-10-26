import numpy

def construct_force_bias_batch(hamiltonian, walker_batch, trial):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walker_batch : class
        walker_batch object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """

    if (walker_batch.name == "SingleDetWalkerBatch" and trial.name == "MultiSlater"):
        return construct_force_bias_batch_single_det(hamiltonian, walker_batch, trial)
    elif (walker_batch.name == "MultiDetTrialWalkerBatch" and trial.name == "MultiSlater"):
        return construct_force_bias_batch_multi_det_trial(hamiltonian, walker_batch, trial)

def construct_force_bias_batch_multi_det_trial(hamiltonian, walker_batch, trial):
    Ga = walker_batch.Ga.reshape(walker_batch.nwalkers, hamiltonian.nbasis**2)
    Gb = walker_batch.Gb.reshape(walker_batch.nwalkers, hamiltonian.nbasis**2)
    # Cholesky vectors. [M^2, nchol]
    # vbias = numpy.dot(hamiltonian.chol_vecs.T, walker.G[0].ravel())
    # vbias += numpy.dot(hamiltonian.chol_vecs.T, walker.G[1].ravel())
    if numpy.isrealobj(hamiltonian.chol_vecs):
        vbias_batch_real = (Ga.real + Gb.real).dot(hamiltonian.chol_vecs)
        vbias_batch_imag = (Ga.imag + Gb.imag).dot(hamiltonian.chol_vecs)
        vbias_batch = numpy.empty((walker_batch.nwalkers, hamiltonian.nchol), dtype=numpy.complex128)
        vbias_batch.real = vbias_batch_real.copy()
        vbias_batch.imag = vbias_batch_imag.copy()
        return vbias_batch
    else:    
        vbias_batch_tmp = (Ga+Gb).dot(hamiltonian.chol_vecs)
        return vbias_batch_tmp

def construct_force_bias_batch_single_det(hamiltonian, walker_batch, trial):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walker_batch : class
        walker_batch object.

    trial : class
        Trial wavefunction object.

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
