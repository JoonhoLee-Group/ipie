import numpy
import scipy.linalg

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


def greens_function_qr_strat(walkers, iw, slice_ix=None, inplace=True):
    """Compute the Green's function for walker with index `iw` at time 
    `slice_ix`. Uses the Stratification method (DOI 10.1109/IPDPS.2012.37)
    """
    stack_iw = walkers.stack[iw]

    if slice_ix == None:
        slice_ix = stack_iw.time_slice

    bin_ix = slice_ix // stack_iw.stack_size
    # For final time slice want first block to be the rightmost (for energy
    # evaluation).
    if bin_ix == stack_iw.nstack:
        bin_ix = -1

    Ga_iw, Gb_iw = None, None
    if not inplace:
        Ga_iw = numpy.zeros(walkers.Ga[iw].shape, walkers.Ga.dtype)
        Gb_iw = numpy.zeros(walkers.Gb[iw].shape, walkers.Gb.dtype)

    for spin in [0, 1]:
        # Need to construct the product A(l) = B_l B_{l-1}..B_L...B_{l+1} in
        # stable way. Iteratively construct column pivoted QR decompositions
        # (A = QDT) starting from the rightmost (product of) propagator(s).
        B = stack_iw.get((bin_ix + 1) % stack_iw.nstack)

        (Q1, R1, P1) = scipy.linalg.qr(B[spin], pivoting=True, check_finite=False)
        # Form D matrices
        D1 = numpy.diag(R1.diagonal())
        D1inv = numpy.diag(1.0 / R1.diagonal())
        T1 = numpy.einsum("ii,ij->ij", D1inv, R1)
        # permute them
        T1[:, P1] = T1[:, range(walkers.nbasis)]

        for i in range(2, stack_iw.nstack + 1):
            ix = (bin_ix + i) % stack_iw.nstack
            B = stack_iw.get(ix)
            C2 = numpy.dot(numpy.dot(B[spin], Q1), D1)
            (Q1, R1, P1) = scipy.linalg.qr(C2, pivoting=True, check_finite=False)
            # Compute D matrices
            D1inv = numpy.diag(1.0 / R1.diagonal())
            D1 = numpy.diag(R1.diagonal())
            tmp = numpy.einsum("ii,ij->ij", D1inv, R1)
            tmp[:, P1] = tmp[:, range(walkers.nbasis)]
            T1 = numpy.dot(tmp, T1)

        # G^{-1} = 1+A = 1+QDT = Q (Q^{-1}T^{-1}+D) T
        # Write D = Db^{-1} Ds
        # Then G^{-1} = Q Db^{-1}(Db Q^{-1}T^{-1}+Ds) T
        Db = numpy.zeros(B[spin].shape, B[spin].dtype)
        Ds = numpy.zeros(B[spin].shape, B[spin].dtype)
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
        if inplace:
            if spin == 0:
                walkers.Ga[iw] = numpy.dot(
                    numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T))
            else:
                walkers.Gb[iw] = numpy.dot(
                    numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T))

        else:
            if spin == 0:
                Ga_iw = numpy.dot(
                    numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T))

            else:
                Gb_iw = numpy.dot(
                    numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T))

    return Ga_iw, Gb_iw
