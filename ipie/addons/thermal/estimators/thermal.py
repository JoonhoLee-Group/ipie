import numpy
import scipy.linalg

def fermi_factor(ek, beta, mu):
    return 1.0 / (numpy.exp(beta * (ek - mu)) + 1.0)

def one_rdm_from_G(G):
    r"""Compute one-particle reduced density matrix from Green's function.

    .. math::
        rho_{ij} = \langle c_{i}^{\dagger} c_{j} \rangle \\
                 = 1 - G_{ji}
    Parameters
    ----------
    G : :class:`numpy.ndarray`
        Thermal Green's function.

    Returns
    -------
    P : :class:`numpy.ndarray`
        Thermal 1RDM.
    """
    I = numpy.identity(G.shape[-1])
    return numpy.array([I - G[0].T, I - G[1].T], dtype=numpy.complex128)

def one_rdm_stable(BT, num_slices):
    nbasis = BT.shape[-1]
    G = []
    for spin in [0, 1]:
        # Need to construct the product A(l) = B_l B_{l-1}..B_L...B_{l+1} in
        # stable way. Iteratively construct column pivoted QR decompositions
        # (A = QDT) starting from the rightmost (product of) propagator(s).
        (Q1, R1, P1) = scipy.linalg.qr(BT[spin], pivoting=True, check_finite=False)
        # Form D matrices
        D1 = numpy.diag(R1.diagonal())
        D1inv = numpy.diag(1.0 / R1.diagonal())
        T1 = numpy.einsum("ii,ij->ij", D1inv, R1)
        # permute them
        T1[:, P1] = T1[:, range(nbasis)]

        for i in range(0, num_slices - 1):
            C2 = numpy.dot(numpy.dot(BT[spin], Q1), D1)
            (Q1, R1, P1) = scipy.linalg.qr(C2, pivoting=True, check_finite=False)
            # Compute D matrices
            D1inv = numpy.diag(1.0 / R1.diagonal())
            D1 = numpy.diag(R1.diagonal())
            tmp = numpy.einsum("ii,ij->ij", D1inv, R1)
            tmp[:, P1] = tmp[:, range(nbasis)]
            T1 = numpy.dot(tmp, T1)
        # G^{-1} = 1+A = 1+QDT = Q (Q^{-1}T^{-1}+D) T
        # Write D = Db^{-1} Ds
        # Then G^{-1} = Q Db^{-1}(Db Q^{-1}T^{-1}+Ds) T
        Db = numpy.zeros(BT[spin].shape, BT[spin].dtype)
        Ds = numpy.zeros(BT[spin].shape, BT[spin].dtype)
        for i in range(Db.shape[0]):
            absDlcr = abs(Db[i, i])
            if absDlcr > 1.0:
                Db[i, i] = 1.0 / absDlcr
                Ds[i, i] = numpy.sign(D1[i, i])
            else:
                Db[i, i] = 1.0
                Ds[i, i] = D1[i, i]

        T1inv = scipy.linalg.inv(T1, check_finite=False)
        # C = (Db Q^{-1}T^{-1}+Ds)
        C = numpy.dot(numpy.einsum("ii,ij->ij", Db, Q1.conj().T), T1inv) + Ds
        Cinv = scipy.linalg.inv(C, check_finite=False)

        # Then G = T^{-1} C^{-1} Db Q^{-1}
        # Q is unitary.
        G.append(numpy.dot(numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T)))
    return one_rdm_from_G(numpy.array(G))

def particle_number(dmat):
    """Compute average particle number from the thermal 1RDM.

    Parameters
    ----------
    dmat : :class:`numpy.ndarray`
        Thermal 1RDM.

    Returns
    -------
    nav : float
        Average particle number.
    """
    nav = dmat[0].trace() + dmat[1].trace()
    return nav

def entropy(beta, mu, H):
    """Compute the entropy.
    """
    assert numpy.linalg.norm(H[0] - H[1]) < 1e-12
    eigs, eigv = numpy.linalg.eigh(H[0])
    p_i = fermi_factor(eigs, beta, mu)
    S = -2.0 * sum(p * numpy.log(p) + (1 - p) * numpy.log(1 - p) for p in p_i)
    # muN = mu * numpy.eye(H[0].shape[-1], dtype=H[0].dtype)
    # rho = numpy.array([scipy.linalg.expm(-beta*(H[0]-muN)),
    # scipy.linalg.expm(-beta*(H[1]-muN))])
    # W = rho[0] + rho[1]
    # W = W / W.trace()
    # logW = -numpy.trace(scipy.linalg.logm(W))
    # S = -numpy.trace(numpy.dot(W,logW))
    return S
