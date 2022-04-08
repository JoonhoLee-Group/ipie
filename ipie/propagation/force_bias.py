import numpy
from ipie.utils.misc import is_cupy

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
    # Why are there so many transposes here?
    if numpy.isrealobj(hamiltonian.chol_vecs):
        vbias_batch = numpy.empty((hamiltonian.nchol,walker_batch.nwalkers), dtype=numpy.complex128)
        vbias_batch.real = hamiltonian.chol_vecs.T.dot(Ga.T.real + Gb.T.real)
        vbias_batch.imag = hamiltonian.chol_vecs.T.dot(Ga.T.imag + Gb.T.imag)
        vbias_batch = vbias_batch.T.copy()
        return vbias_batch
    else:
        vbias_batch_tmp = hamiltonian.chol_vecs.T.dot(Ga.T+Gb.T)
        vbias_batch_tmp = vbias_batch_tmp.T.copy()
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
    if is_cupy(trial.psi): # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy
        assert(cupy.is_available())
        isrealobj = cupy.isrealobj
        empty = cupy.empty
    else:
        isrealobj = numpy.isrealobj
        empty = numpy.empty


    if (walker_batch.rhf):
        Ghalfa = walker_batch.Ghalfa.reshape(walker_batch.nwalkers, walker_batch.nup*hamiltonian.nbasis)
        if isrealobj(trial._rchola) and isrealobj(trial._rcholb):
            vbias_batch_real = 2.*trial._rchola.dot(Ghalfa.T.real)
            vbias_batch_imag = 2.*trial._rchola.dot(Ghalfa.T.imag)
            vbias_batch = empty((walker_batch.nwalkers, hamiltonian.nchol), dtype=numpy.complex128)
            vbias_batch.real = vbias_batch_real.T.copy()
            vbias_batch.imag = vbias_batch_imag.T.copy()
            return vbias_batch
        else:    
            vbias_batch_tmp = 2.*trial._rchola.dot(Ghalfa.T) 
            return vbias_batch_tmp.T

    else:
        Ghalfa = walker_batch.Ghalfa.reshape(walker_batch.nwalkers, walker_batch.nup*hamiltonian.nbasis)
        Ghalfb = walker_batch.Ghalfb.reshape(walker_batch.nwalkers, walker_batch.ndown*hamiltonian.nbasis)
        if isrealobj(trial._rchola) and isrealobj(trial._rcholb):
            vbias_batch_real = trial._rchola.dot(Ghalfa.T.real) + trial._rcholb.dot(Ghalfb.T.real)
            vbias_batch_imag = trial._rchola.dot(Ghalfa.T.imag) + trial._rcholb.dot(Ghalfb.T.imag)
            vbias_batch = empty((walker_batch.nwalkers, hamiltonian.nchol), dtype=numpy.complex128)
            vbias_batch.real = vbias_batch_real.T.copy()
            vbias_batch.imag = vbias_batch_imag.T.copy()
            return vbias_batch
        else:    
            vbias_batch_tmp = trial._rchola.dot(Ghalfa.T) + trial._rcholb.dot(Ghalfb.T)
            return vbias_batch_tmp.T
