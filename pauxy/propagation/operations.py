import numpy


def propagate_single(psi, system, B):
    r"""Perform backpropagation for single configuration.

    Deals with GHF and RHF/UHF walkers.

    Parameters
    ---------
    phi : walker object
        Walker.
    system : system object in general.
        Container for model input options.
    B : :class:`numpy.ndarray`
        Propagator matrix.
    """
    nup = system.nup
    if len(B.shape) == 3:
        psi[:,:nup] = B[0].dot(psi[:,:nup])
        psi[:,nup:] = B[1].dot(psi[:,nup:])
    else:
        M = system.nbasis
        psi[:M,:nup] = B[:M,:M].dot(psi[:M,:nup])
        psi[M:,nup:] = B[M:,M:].dot(psi[M:,nup:])


# TODO: Rename this
def kinetic_real(phi, system, bt2, H1diag=False):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the continuus algorithm and free propagation.

    todo : this is the same a propagating by an arbitrary matrix, remove.

    Parameters
    ----------
    walker : :class:`pauxy.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pauxy.state.State`
        Simulation state.
    """
    nup = system.nup
    # Assuming that our walker is in UHF form.
    if (H1diag):
        phi[:,:nup] = numpy.einsum("ii,ij->ij", bt2[0],phi[:,:nup])
        phi[:,nup:] = numpy.einsum("ii,ij->ij", bt2[1],phi[:,nup:])
    else:
        phi[:,:nup] = bt2[0].dot(phi[:,:nup])
        phi[:,nup:] = bt2[1].dot(phi[:,nup:])

def kinetic_real_stochastic(phi, system, bt2, nsamples, H1diag=False):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the continuus algorithm and free propagation.

    todo : this is the same a propagating by an arbitrary matrix, remove.

    Parameters
    ----------
    walker : :class:`pauxy.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pauxy.state.State`
        Simulation state.
    """
    nup = system.nup
    if (H1diag): # if diagonal then it's done exactly
        phi[:,:nup] = numpy.einsum("ii,ij->ij", bt2[0],phi[:,:nup])
        phi[:,nup:] = numpy.einsum("ii,ij->ij", bt2[1],phi[:,nup:])
    else:
        nbasis = phi.shape[0]

        theta = numpy.zeros((nbasis,nsamples), dtype=numpy.int64)
        for i in range(nsamples):
            theta[:,i] = (2*numpy.random.randint(0,2,size=(nbasis))-1)

        # iden = theta.dot(theta.T) * (1./nsamples)
        tmp = bt2[0].dot(theta)
        tmp2 = theta.T.dot(phi[:,:nup])
        phi[:,:nup] = tmp.dot(tmp2) * (1./nsamples)

        tmp = bt2[1].dot(theta)
        tmp2 = theta.T.dot(phi[:,nup:])
        phi[:,nup:] = tmp.dot(tmp2) * (1./nsamples)



def local_energy_bound(local_energy, mean, threshold):
    """Try to suppress rare population events by imposing local energy bound.

    See: Purwanto et al., Phys. Rev. B 80, 214116 (2009).

    Parameters
    ----------
    local_energy : float
        Local energy of current walker
    mean : float
        Mean value of local energy about which we impose the threshold / bound.
    threshold : float
        Amount of lee-way for energy fluctuations about the mean.
    """

    maximum = mean + threshold
    minimum = mean - threshold

    if (local_energy >= maximum):
        local_energy = maximum
    elif (local_energy < minimum):
        local_energy = minimum
    else:
        local_energy = local_energy

    return local_energy

def kinetic_ghf(phi, system, bt2):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the GHF trial wavefunction.

    Parameters
    ----------
    walker : :class:`pauxy.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pauxy.state.State`
        Simulation state.
    """
    nup = system.nup
    nb = system.nbasis
    # Assuming that our walker is in GHF form.
    phi[:nb,:nup] = bt2.dot(phi[:nb,:nup])
    phi[nb:,nup:] = bt2.dot(phi[nb:,nup:])


def propagate_potential_auxf(phi, state, field_config):
    """Propagate walker given a fixed set of auxiliary fields.

    Useful for debugging.

    Parameters
    ----------
    phi : :class:`numpy.ndarray`
        Walker's slater determinant to be updated.
    state : :class:`pauxy.state.State`
        Simulation state.
    field_config : numpy array
        Auxiliary field configurations to apply to walker.
    """

    bv_up = numpy.array([state.auxf[xi, 0] for xi in field_config])
    bv_down = numpy.array([state.auxf[xi, 1] for xi in field_config])
    phi[:,:nup] = numpy.einsum('i,ij->ij', bv_up, phi[:,:nup])
    phi[:,nup:] = numpy.einsum('i,ij->ij', bv_down, phi[:,nup:])
