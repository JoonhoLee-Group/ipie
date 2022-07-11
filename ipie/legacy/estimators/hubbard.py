import numpy


def local_energy_hubbard_holstein_momentum(ham, G, P, Lap, Ghalf=None):
    r"""Calculate local energy of walker for the Hubbard-Hostein model.

    Parameters
    ----------
    ham : :class:`HubbardHolstein`
        ham information for the HubbardHolstein model.
    G : :class:`numpy.ndarray`
        Walker's "Green's function"

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    # T = kinetic_lang_firsov(ham.t, ham.gamma_lf, P, ham.nx, ham.ny, ham.ktwist)

    Dp = numpy.array([numpy.exp(1j * ham.gamma_lf * P[i]) for i in range(ham.nbasis)])
    T = numpy.zeros_like(ham.T, dtype=numpy.complex128)
    T[0] = numpy.diag(Dp).dot(ham.T[0]).dot(numpy.diag(Dp.T.conj()))
    T[1] = numpy.diag(Dp).dot(ham.T[1]).dot(numpy.diag(Dp.T.conj()))

    ke = numpy.sum(T[0] * G[0] + T[1] * G[1])

    sqrttwomw = numpy.sqrt(2.0 * ham.m * ham.w0)
    assert ham.gamma_lf * ham.w0 == ham.g * sqrttwomw

    Ueff = ham.U + ham.gamma_lf**2 * ham.w0 - 2.0 * ham.g * ham.gamma_lf * sqrttwomw

    if ham.symmetric:
        pe = -0.5 * Ueff * (G[0].trace() + G[1].trace())

    pe = Ueff * numpy.dot(G[0].diagonal(), G[1].diagonal())

    pe_ph = -0.5 * ham.w0**2 * ham.m * numpy.sum(Lap)
    ke_ph = 0.5 * numpy.sum(P * P) / ham.m - 0.5 * ham.w0 * ham.nbasis

    rho = G[0].diagonal() + G[1].diagonal()

    e_eph = (
        ham.gamma_lf**2 * ham.w0 / 2.0 - ham.g * ham.gamma_lf * sqrttwomw
    ) * numpy.sum(rho)

    etot = ke + pe + pe_ph + ke_ph + e_eph

    Eph = ke_ph + pe_ph
    Eel = ke + pe
    Eeb = e_eph

    return (etot, ke + pe, ke_ph + pe_ph + e_eph)


def local_energy_hubbard_holstein(ham, G, X, Lap, Ghalf=None):
    r"""Calculate local energy of walker for the Hubbard-Hostein model.

    Parameters
    ----------
    ham : :class:`HubbardHolstein`
        ham information for the HubbardHolstein model.
    G : :class:`numpy.ndarray`
        Walker's "Green's function"
    X : :class:`numpy.ndarray`
        Walker's phonon coordinate

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.sum(ham.T[0] * G[0] + ham.T[1] * G[1])

    if ham.symmetric:
        pe = -0.5 * ham.U * (G[0].trace() + G[1].trace())

    pe = ham.U * numpy.dot(G[0].diagonal(), G[1].diagonal())

    pe_ph = 0.5 * ham.w0**2 * ham.m * numpy.sum(X * X)

    ke_ph = -0.5 * numpy.sum(Lap) / ham.m - 0.5 * ham.w0 * ham.nbasis

    rho = G[0].diagonal() + G[1].diagonal()
    e_eph = -ham.g * numpy.sqrt(ham.m * ham.w0 * 2.0) * numpy.dot(rho, X)

    etot = ke + pe + pe_ph + ke_ph + e_eph

    Eph = ke_ph + pe_ph
    Eel = ke + pe
    Eeb = e_eph

    return (etot, ke + pe, ke_ph + pe_ph + e_eph)


def local_energy_hubbard(ham, G, Ghalf=None):
    r"""Calculate local energy of walker for the Hubbard model.

    Parameters
    ----------
    ham : :class:`Hubbard`
        ham information for the Hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "Green's function"

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.sum(ham.T[0] * G[0] + ham.T[1] * G[1])
    # Todo: Stupid
    if ham.symmetric:
        pe = -0.5 * ham.U * (G[0].trace() + G[1].trace())
    pe = ham.U * numpy.dot(G[0].diagonal(), G[1].diagonal())

    return (ke + pe, ke, pe)


def local_energy_hubbard_ghf(ham, Gi, weights, denom):
    """Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    ham : :class:`Hubbard`
        ham information for the Hubbard model.
    Gi : :class:`numpy.ndarray`
        Array of Walker's "Green's function"
    denom : float
        Overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    ke = numpy.einsum("i,ikl,kl->", weights, Gi, ham.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(Gi[:, : ham.nbasis, : ham.nbasis], axis1=1, axis2=2)
    gdd = numpy.diagonal(Gi[:, ham.nbasis :, ham.nbasis :], axis1=1, axis2=2)
    gud = numpy.diagonal(Gi[:, ham.nbasis :, : ham.nbasis], axis1=1, axis2=2)
    gdu = numpy.diagonal(Gi[:, : ham.nbasis, ham.nbasis :], axis1=1, axis2=2)
    gdiag = guu * gdd - gud * gdu
    pe = ham.U * numpy.einsum("j,jk->", weights, gdiag) / denom
    return (ke + pe, ke, pe)


def local_energy_hubbard_ghf_full(ham, GAB, weights):
    r"""Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    ham : :class:`Hubbard`
        ham information for the Hubbard model.
    GAB : :class:`numpy.ndarray`
        Matrix of Green's functions for different SDs A and B.
    weights : :class:`numpy.ndarray`
        Components of overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L, T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    denom = numpy.sum(weights)
    ke = numpy.einsum("ij,ijkl,kl->", weights, GAB, ham.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(GAB[:, :, : ham.nbasis, : ham.nbasis], axis1=2, axis2=3)
    gdd = numpy.diagonal(GAB[:, :, ham.nbasis :, ham.nbasis :], axis1=2, axis2=3)
    gud = numpy.diagonal(GAB[:, :, ham.nbasis :, : ham.nbasis], axis1=2, axis2=3)
    gdu = numpy.diagonal(GAB[:, :, : ham.nbasis, ham.nbasis :], axis1=2, axis2=3)
    gdiag = guu * gdd - gud * gdu
    pe = ham.U * numpy.einsum("ij,ijk->", weights, gdiag) / denom
    return (ke + pe, ke, pe)


def local_energy_multi_det(ham, Gi, weights):
    """Calculate local energy of GHF walker for the Hubbard model.

    Parameters
    ----------
    ham : :class:`Hubbard`
        ham information for the Hubbard model.
    Gi : :class:`numpy.ndarray`
        Array of Walker's "Green's function"
    weights : :class:`numpy.ndarray`
        Components of overlap of trial wavefunction with walker.

    Returns
    -------
    (E_L(phi), T, V): tuple
        Local, kinetic and potential energies of given walker phi.
    """
    denom = numpy.sum(weights)
    ke = numpy.einsum("i,ikl,kl->", weights, Gi, ham.Text) / denom
    # numpy.diagonal returns a view so there should be no overhead in creating
    # temporary arrays.
    guu = numpy.diagonal(Gi[:, :, : ham.nup], axis1=1, axis2=2)
    gdd = numpy.diagonal(Gi[:, :, ham.nup :], axis1=1, axis2=2)
    pe = ham.U * numpy.einsum("j,jk->", weights, guu * gdd) / denom
    return (ke + pe, ke, pe)


def fock_hubbard(ham, P):
    """Hubbard Fock Matrix
    F_{ij} = T_{ij} + U(<niu>nid + <nid>niu)_{ij}
    """
    niu = numpy.diag(P[0].diagonal())
    nid = numpy.diag(P[1].diagonal())
    return ham.T + ham.U * numpy.array([nid, niu])
