import numpy
import scipy.linalg


def fermi_factor(ek, beta, mu):
    return 1.0 / (numpy.exp(beta * (ek - mu)) + 1.0)


def greens_function_unstable(A):
    r"""Construct Green's function from density matrix.

    .. math::
        G_{ij} = \langle c_{i} c_j^{\dagger} \rangle \\
               = \left[\frac{1}{1+A}\right]_{ij}

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Density matrix (product of B matrices).

    Returns
    -------
    G : :class:`numpy.ndarray`
        Thermal Green's function.
    """
    I = numpy.identity(A.shape[-1])
    return numpy.array([scipy.linalg.inv(I + A[0]), scipy.linalg.inv(I + A[1])])


def greens_function(A):
    r"""Construct Greens function from density matrix.

    .. math::
        G_{ij} = \langle c_{i} c_j^{\dagger} \rangle \\
               = \left[\frac{1}{1+A}\right]_{ij}

    Uses stable algorithm from White et al. (1988)

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Density matrix (product of B matrices).

    Returns
    -------
    G : :class:`numpy.ndarray`
        Thermal Green's function.
    """
    G = numpy.zeros(A.shape, dtype=A.dtype)
    (U1, S1, V1) = scipy.linalg.svd(A)
    T = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
    (U2, S2, V2) = scipy.linalg.svd(T)
    U3 = numpy.dot(U1, U2)
    D3 = numpy.diag(1.0 / S2)
    V3 = numpy.dot(V2, V1)
    G = (V3.conj().T).dot(D3).dot(U3.conj().T)
    return G


def inverse_greens_function(A):
    """Inverse greens function from A"""

    Ginv = numpy.zeros(A.shape, dtype=A.dtype)
    (U1, S1, V1) = scipy.linalg.svd(A)
    T = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
    (U2, S2, V2) = scipy.linalg.svd(T)
    U3 = numpy.dot(U1, U2)
    D3 = numpy.diag(S2)
    V3 = numpy.dot(V2, V1)
    Ginv = (V3.conj().T).dot(D3).dot(U3.conj().T)
    return Ginv


def inverse_greens_function_qr(A):
    """Inverse greens function from A"""

    Ginv = numpy.zeros(A.shape, dtype=A.dtype)

    (U1, V1) = scipy.linalg.qr(A, pivoting=False)
    V1inv = scipy.linalg.solve_triangular(V1, numpy.identity(V1.shape[0]))
    T = numpy.dot(U1.conj().T, V1inv) + numpy.identity(V1.shape[0])
    (U2, V2) = scipy.linalg.qr(T, pivoting=False)
    U3 = numpy.dot(U1, U2)
    V3 = numpy.dot(V2, V1)
    Ginv = U3.dot(V3)
    # (U1,S1,V1) = scipy.linalg.svd(A)
    # T = numpy.dot(U1.conj().T, V1.conj().T) + numpy.diag(S1)
    # (U2,S2,V2) = scipy.linalg.svd(T)
    # U3 = numpy.dot(U1, U2)
    # D3 = numpy.diag(S2)
    # V3 = numpy.dot(V2, V1)
    # Ginv = (V3.conj().T).dot(D3).dot(U3.conj().T)
    return Ginv


def one_rdm(A):
    r"""Compute one-particle reduced density matrix

    .. math::
        rho_{ij} = \langle c_{i}^{\dagger} c_{j} \rangle \\
                 = 1 - G_{ji}
    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Density matrix (product of B matrices).

    Returns
    -------
    P : :class:`numpy.ndarray`
        Thermal 1RDM.
    """
    I = numpy.identity(A.shape[-1])
    G = numpy.array([greens_function(A[0]), greens_function(A[1])])
    return numpy.array([I - G[0].T, I - G[1].T])


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


def particle_number(dmat):
    """Compute average particle number.

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
        G.append(
            numpy.dot(
                numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T)
            )
        )
    return one_rdm_from_G(numpy.array(G))


def entropy(beta, mu, H):
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
