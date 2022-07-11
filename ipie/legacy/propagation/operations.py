import numpy

from ipie.utils.misc import is_cupy


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
        psi[:, :nup] = B[0].dot(psi[:, :nup])
        psi[:, nup:] = B[1].dot(psi[:, nup:])
    else:
        M = system.nbasis
        psi[:M, :nup] = B[:M, :M].dot(psi[:M, :nup])
        psi[M:, nup:] = B[M:, M:].dot(psi[M:, nup:])


# TODO: Rename this
def kinetic_real(phi, system, bt2, H1diag=False):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the continuus algorithm and free propagation.

    todo : this is the same a propagating by an arbitrary matrix, remove.

    Parameters
    ----------
    walker : :class:`pie.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pie.state.State`
        Simulation state.
    """
    if is_cupy(bt2[0]):
        import cupy

        assert cupy.is_available()
        einsum = cupy.einsum
    else:
        einsum = numpy.einsum
    nup = system.nup
    # Assuming that our walker is in UHF form.
    if H1diag:
        phi[:, :nup] = einsum("ii,ij->ij", bt2[0], phi[:, :nup])
        phi[:, nup:] = einsum("ii,ij->ij", bt2[1], phi[:, nup:])
    else:
        phi[:, :nup] = bt2[0].dot(phi[:, :nup])
        phi[:, nup:] = bt2[1].dot(phi[:, nup:])

    if is_cupy(
        bt2[0]
    ):  # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy

        cupy.cuda.stream.get_current_stream().synchronize()

    return


def kinetic_spin_real_batch(phi, bt2, H1diag=False):
    r"""Propagate by the kinetic term by direct matrix multiplication. Only one spin component. Assuming phi is a batch.

    For use with the continuus algorithm and free propagation.

    todo : this is the same a propagating by an arbitrary matrix, remove.

    Parameters
    ----------
    walker : :class:`pie.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pie.state.State`
        Simulation state.
    """
    if is_cupy(bt2):
        import cupy

        assert cupy.is_available()
        einsum = cupy.einsum
    else:
        einsum = numpy.einsum

    # Assuming that our walker is in UHF form.
    if H1diag:
        phi[:, :] = einsum("ii,wij->ij", bt2, phi)
    else:
        if is_cupy(bt2):
            phi = einsum("ik,wkj->wij", bt2, phi, optimize=True)
        else:
            # Loop is O(10x) times faster on CPU for FeP benchmark
            for iw in range(phi.shape[0]):
                phi[iw] = numpy.dot(bt2, phi[iw])

    return phi


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

    if local_energy >= maximum:
        local_energy = maximum
    elif local_energy < minimum:
        local_energy = minimum
    else:
        local_energy = local_energy

    return local_energy


def kinetic_ghf(phi, system, bt2):
    r"""Propagate by the kinetic term by direct matrix multiplication.

    For use with the GHF trial wavefunction.

    Parameters
    ----------
    walker : :class:`pie.walker.Walker`
        Walker object to be updated. on output we have acted on
        :math:`|\phi_i\rangle` by :math:`B_{T/2}` and updated the weight
        appropriately.  updates inplace.
    state : :class:`pie.state.State`
        Simulation state.
    """
    nup = system.nup
    nb = system.nbasis
    # Assuming that our walker is in GHF form.
    phi[:nb, :nup] = bt2.dot(phi[:nb, :nup])
    phi[nb:, nup:] = bt2.dot(phi[nb:, nup:])


def propagate_potential_auxf(phi, state, field_config):
    """Propagate walker given a fixed set of auxiliary fields.

    Useful for debugging.

    Parameters
    ----------
    phi : :class:`numpy.ndarray`
        Walker's slater determinant to be updated.
    state : :class:`pie.state.State`
        Simulation state.
    field_config : numpy array
        Auxiliary field configurations to apply to walker.
    """

    bv_up = numpy.array([state.auxf[xi, 0] for xi in field_config])
    bv_down = numpy.array([state.auxf[xi, 1] for xi in field_config])
    phi[:, :nup] = numpy.einsum("i,ij->ij", bv_up, phi[:, :nup])
    phi[:, nup:] = numpy.einsum("i,ij->ij", bv_down, phi[:, nup:])
