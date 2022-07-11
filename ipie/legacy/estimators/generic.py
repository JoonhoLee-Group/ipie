import sys

import numpy

from ipie.utils.misc import is_cupy


def local_energy_generic(h1e, eri, G, ecore=0.0, Ghalf=None):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the full form for the two-electron integrals.

    For testing purposes only.

    Parameters
    ----------
    system : :class:`hubbard`
        System information for the hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "green's function"

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    e1 = numpy.einsum("ij,ij->", h1e[0], G[0]) + numpy.einsum("ij,ij->", h1e[1], G[1])
    euu = 0.5 * (
        numpy.einsum("prqs,pr,qs->", eri, G[0], G[0])
        - numpy.einsum("prqs,ps,qr->", eri, G[0], G[0])
    )
    edd = 0.5 * (
        numpy.einsum("prqs,pr,qs->", eri, G[1], G[1])
        - numpy.einsum("prqs,ps,qr->", eri, G[1], G[1])
    )
    eud = 0.5 * numpy.einsum("prqs,pr,qs->", eri, G[0], G[1])
    edu = 0.5 * numpy.einsum("prqs,pr,qs->", eri, G[1], G[0])
    e2 = euu + edd + eud + edu
    return (e1 + e2 + ecore, e1 + ecore, e2)


def local_energy_generic_pno(
    system,
    G,
    Ghalf=None,
    eri=None,
    C0=None,
    ecoul0=None,
    exxa0=None,
    exxb0=None,
    UVT=None,
):

    na = system.nup
    nb = system.ndown
    M = system.nbasis

    UVT_aa = UVT[0]
    UVT_bb = UVT[1]
    UVT_ab = UVT[2]

    Ga, Gb = Ghalf[0], Ghalf[1]

    # Element wise multiplication.
    e1b = numpy.sum(system.H1[0] * G[0]) + numpy.sum(system.H1[1] * G[1])

    eJaa = 0.0
    eKaa = 0.0

    if len(C0.shape) == 3:
        CT = C0[0, :, :].T
    else:
        CT = C0[:, :].T

    GTa = CT[:na, :]  # hard-coded to do single slater
    GTb = CT[na:, :]  # hard-coded to do single slater

    for (i, j), (U, VT) in zip(system.ij_list_aa, UVT_aa):

        if i == j:
            c = 0.5
        else:
            c = 1.0

        theta_i = Ga[i, :]
        theta_j = Ga[j, :]

        thetaT_i = GTa[i, :]
        thetaT_j = GTa[j, :]

        thetaU = numpy.einsum("p,pk->k", theta_i, U)
        thetaV = numpy.einsum("p,kp->k", theta_j, VT)

        thetaTU = numpy.einsum("p,pk->k", thetaT_i, U)
        thetaTV = numpy.einsum("p,kp->k", thetaT_j, VT)

        eJaa += c * (numpy.dot(thetaU, thetaV) - numpy.dot(thetaTU, thetaTV))

        thetaU = numpy.einsum("p,pk->k", theta_j, U)
        thetaV = numpy.einsum("p,kp->k", theta_i, VT)
        thetaTU = numpy.einsum("p,pk->k", thetaT_j, U)
        thetaTV = numpy.einsum("p,kp->k", thetaT_i, VT)
        eKaa -= c * (numpy.dot(thetaU, thetaV) - numpy.dot(thetaTU, thetaTV))

    eJbb = 0.0
    eKbb = 0.0

    for (i, j), (U, VT) in zip(system.ij_list_bb, UVT_bb):
        if i == j:
            c = 0.5
        else:
            c = 1.0

        theta_i = Gb[i, :]
        theta_j = Gb[j, :]
        thetaT_i = GTb[i, :]
        thetaT_j = GTb[j, :]

        thetaU = numpy.einsum("p,pk->k", theta_i, U)
        thetaV = numpy.einsum("p,kp->k", theta_j, VT)
        thetaTU = numpy.einsum("p,pk->k", thetaT_i, U)
        thetaTV = numpy.einsum("p,kp->k", thetaT_j, VT)
        eJbb += c * (numpy.dot(thetaU, thetaV) - numpy.dot(thetaTU, thetaTV))

        thetaU = numpy.einsum("p,pk->k", theta_j, U)
        thetaV = numpy.einsum("p,kp->k", theta_i, VT)
        thetaTU = numpy.einsum("p,pk->k", thetaT_j, U)
        thetaTV = numpy.einsum("p,kp->k", thetaT_i, VT)
        eKbb -= c * (numpy.dot(thetaU, thetaV) - numpy.dot(thetaTU, thetaTV))

    eJab = 0.0
    for (i, j), (U, VT) in zip(system.ij_list_ab, UVT_ab):
        theta_i = Ga[i, :]
        theta_j = Gb[j, :]
        thetaT_i = GTa[i, :]
        thetaT_j = GTb[j, :]
        thetaU = numpy.einsum("p,pk->k", theta_i, U)
        thetaV = numpy.einsum("p,kp->k", theta_j, VT)
        thetaTU = numpy.einsum("p,pk->k", thetaT_i, U)
        thetaTV = numpy.einsum("p,kp->k", thetaT_j, VT)
        eJab += numpy.dot(thetaU, thetaV) - numpy.dot(thetaTU, thetaTV)

    e2b = 0.5 * (ecoul0 - exxa0 - exxb0) + eJaa + eJbb + eJab + eKaa + eKbb

    return (e1b + e2b + system.ecore, e1b + system.ecore, e2b)


def _exx_compute_batch(rchol_a, rchol_b, GaT_stacked, GbT_stacked, lwalker):
    """
    Internal function for computing exchange two-electron integral energy
    of batched walkers. The stacked batching ends up being about 30% faster
    than simple loop over walkers.

    Parameters
    ----------
    rchol_a: :class:`numpy.ndarray`
        alpha-spin half-rotated cholesky vectors that are (naux, nalpha, nbasis)
    rchol_b: :class:`numpy.ndarray`
        beta-spin half-rotated cholesky vectors that are (naux, nbeta, nbasis)
    GaT_stacked: :class:`numpy.ndarray`
        alpha-spin half-rotated Greens function of size (nbasis, nalpha * nwalker)
    GbT_stacked: :class:`numpy.ndarray`
        beta-spin half-rotated Greens function of size (nbasis, nbeta * nwalker)
    Returns
    -------
    exx: numpy.ndarary
        vector of exchange contributions for each walker
    """
    naux = rchol_a.shape[0]
    nbasis = GaT_stacked.shape[0]
    nalpha = GaT_stacked.shape[1] // lwalker
    nbeta = GbT_stacked.shape[1] // lwalker

    exx_vec_a = numpy.zeros(lwalker, dtype=numpy.complex128)
    exx_vec_b = numpy.zeros(lwalker, dtype=numpy.complex128)

    # Ta = numpy.zeros((nalpha, nalpha * lwalker), dtype=numpy.complex128)
    # Tb = numpy.zeros((nbeta, nbeta * lwalker), dtype=numpy.complex128)

    # Writing this way so in the future we can vmap of naux index of rchol_a
    for x in range(naux):
        rmi_a = rchol_a[x].reshape((nalpha, nbasis))  # can we get rid of this?
        Ta = rmi_a.dot(GaT_stacked)  # (na, na x nwalker)
        # Ta = rmi_a.real.dot(GaT_stacked.real) + 1j * rmi_a.real.dot(GaT_stacked.imag)
        rmi_b = rchol_b[x].reshape((nbeta, nbasis))
        Tb = rmi_b.dot(GbT_stacked)  # (nb, nb x nwalker)
        # Tb = rmi_b.real.dot(GbT_stacked.real) + 1j * rmi_b.real.dot(GbT_stacked.imag)
        Ta = Ta.reshape((nalpha, lwalker, nalpha))  # reshape into 3-tensor for tdot
        Tb = Tb.reshape((nbeta, lwalker, nbeta))
        exx_vec_a += numpy.einsum("ikj,jki->k", Ta, Ta, optimize=True)
        exx_vec_b += numpy.einsum("ikj,jki->k", Tb, Tb, optimize=True)
    return exx_vec_b + exx_vec_a


def local_energy_generic_cholesky_opt_batched(
    system,
    ham,
    Ga_batch: numpy.ndarray,
    Gb_batch: numpy.ndarray,
    Ghalfa_batch: numpy.ndarray,
    Ghalfb_batch: numpy.ndarray,
    rchola: numpy.ndarray,
    rcholb: numpy.ndarray,
):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals. and batched
    walkers.  The intended use is CPU batching where walker G functions are
    stacked so we can use gemm on bigger arrays.

    Parameters
    ----------
    system : :class:`Generic`
        System information for Generic.
    ham : :class:`Abinitio`
        Contains necessary hamiltonian information
    Ga_batched : :class:`numpy.ndarray`
        alpha-spin Walker's "green's function" 3-tensor (nwalker, nbasis, nbasis)
    Gb_batched : :class:`numpy.ndarray`
        beta-spin Walker's "green's function" 3-tensor (nwalker, nbasis, nbasis)
    Ghalfa_batched : :class:`numpy.ndarray`
        alpha-spin Walker's half-rotated "green's function" 3-tensor (nwalker, nalpha, nbasis)
    Ghalfb_batched : :class:`numpy.ndarray`
        beta-spin Walker's half-rotated "green's function" 3-tensor (nwalker, nbeta, nbasis)
    rchola : :class:`numpy.ndarray`
        alpha-spin trial's half-rotated choleksy vectors (naux, nalpha * nbasis)
    rcholb : :class:`numpy.ndarray`
        beta-spin trial's half-rotated choleksy vectors (naux, nbeta * nbasis)

    Returns
    -------
    (E, T, V): tuple of vectors
        vectors of Local, kinetic and potential energies for each walker
    """
    # Element wise multiplication.
    nwalker = Ga_batch.shape[0]
    e1_vec = numpy.zeros(nwalker, dtype=numpy.complex128)
    ecoul_vec = numpy.zeros(nwalker, dtype=numpy.complex128)
    # simple loop because this part isn't the slow bit
    for widx in range(nwalker):
        e1b = numpy.sum(ham.H1[0] * Ga_batch[widx]) + numpy.sum(
            ham.H1[1] * Gb_batch[widx]
        )
        e1_vec[widx] = e1b
        nalpha, nbeta = system.nup, system.ndown
        nbasis = ham.nbasis
        if rchola is not None:
            naux = rchola.shape[0]

        Xa = rchola.dot(Ghalfa_batch[widx].ravel())
        Xb = rcholb.dot(Ghalfb_batch[widx].ravel())
        ecoul = numpy.dot(Xa, Xa)
        ecoul += numpy.dot(Xb, Xb)
        ecoul += 2 * numpy.dot(Xa, Xb)
        ecoul_vec[widx] = ecoul

    # transpose batch of walkers as exx prep
    GhalfaT_stacked = numpy.hstack([*Ghalfa_batch.transpose((0, 2, 1)).copy()])
    GhalfbT_stacked = numpy.hstack([*Ghalfb_batch.transpose((0, 2, 1)).copy()])
    # call batched exx computation
    exx_vec = _exx_compute_batch(
        rchol_a=rchola,
        rchol_b=rcholb,
        GaT_stacked=GhalfaT_stacked,
        GbT_stacked=GhalfbT_stacked,
        lwalker=nwalker,
    )
    e2b_vec = 0.5 * (ecoul_vec - exx_vec)

    return (e1_vec + e2b_vec + ham.ecore, e1_vec + ham.ecore, e2b_vec)


def local_energy_generic_cholesky(system, ham, G, Ghalf=None):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals.

    Parameters
    ----------
    system : :class:`Generic`
        generic system information
    ham : :class:`Generic`
        ab-initio hamiltonian information
    G : :class:`numpy.ndarray`
        Walker's "green's function"

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    # Element wise multiplication.
    e1b = numpy.sum(ham.H1[0] * G[0]) + numpy.sum(ham.H1[1] * G[1])
    nalpha, nbeta = system.nup, system.ndown
    nbasis = ham.nbasis
    nchol = ham.nchol
    Ga, Gb = G[0], G[1]

    # Xa = numpy.dot(ham.chol_vecs, Ga.ravel())
    # Xb = numpy.dot(ham.chol_vecs, Gb.ravel())

    if numpy.isrealobj(ham.chol_vecs):
        # Xa = ham.chol_vecs.T.dot(Ga.real.ravel()) + 1.j * ham.chol_vecs.dot(Ga.imag.ravel())
        # Xb = ham.chol_vecs.T.dot(Gb.real.ravel()) + 1.j * ham.chol_vecs.dot(Gb.imag.ravel())
        Xa = ham.chol_vecs.T.dot(Ga.real.ravel()) + 1.0j * ham.chol_vecs.T.dot(
            Ga.imag.ravel()
        )
        Xb = ham.chol_vecs.T.dot(Gb.real.ravel()) + 1.0j * ham.chol_vecs.T.dot(
            Gb.imag.ravel()
        )
    else:
        Xa = ham.chol_vecs.T.dot(Ga.ravel())
        Xb = ham.chol_vecs.T.dot(Gb.ravel())

    ecoul = numpy.dot(Xa, Xa)
    ecoul += numpy.dot(Xb, Xb)
    ecoul += 2 * numpy.dot(Xa, Xb)

    # T[l,k,n] = \sum_i L[i,k,n] G[i,l]
    # exx  = \sum_{nlk} T[l,k,n] T[k,l,n]
    # cv = ham.chol_vecs.T.reshape((nbasis,nbasis,-1))
    # Ta = numpy.tensordot(Ga, cv, axes=((0),(0)))
    # exxa = numpy.tensordot(Ta, Ta, axes=((0,1,2),(1,0,2)))
    # Tb = numpy.tensordot(Gb, cv, axes=((0),(0)))
    # exxb = numpy.tensordot(Tb, Tb, axes=((0,1,2),(1,0,2)))
    # exx = exxa + exxb
    # e2b = 0.5 * (ecoul - exx)

    T = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)

    GaT = Ga.T.copy()
    GbT = Gb.T.copy()

    exx = 0.0j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    if numpy.isrealobj(ham.chol_vecs):
        for x in range(nchol):  # write a cython function that calls blas for this.
            Lmn = ham.chol_vecs[:, x].reshape((nbasis, nbasis))
            T[:, :].real = GaT.real.dot(Lmn)
            T[:, :].imag = GaT.imag.dot(Lmn)
            exx += numpy.trace(T.dot(T))
            T[:, :].real = GbT.real.dot(Lmn)
            T[:, :].imag = GbT.imag.dot(Lmn)
            exx += numpy.trace(T.dot(T))
    else:
        for x in range(nchol):  # write a cython function that calls blas for this.
            Lmn = ham.chol_vecs[:, x].reshape((nbasis, nbasis))
            T[:, :] = GaT.dot(Lmn)
            exx += numpy.trace(T.dot(T))
            T[:, :] = GbT.dot(Lmn)
            exx += numpy.trace(T.dot(T))

    e2b = 0.5 * (ecoul - exx)

    return (e1b + e2b + ham.ecore, e1b + ham.ecore, e2b)


def local_energy_generic_cholesky_opt_stochastic(
    system, G, nsamples, Ghalf, rchol=None, C0=None, ecoul0=None, exxa0=None, exxb0=None
):
    r"""Calculate local for generic two-body hamiltonian.
    This uses the cholesky decomposed two-electron integrals.
    Parameters
    ----------
    system : :class:`hubbard`
        System information for the hubbard model.
    G : :class:`numpy.ndarray`
        Walker's "green's function"
    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    # import cProfile
    # pr = cProfile.Profile()
    # pr.enable()

    if type(C0) == numpy.ndarray:
        control = True
    else:
        control = False

    # Element wise multiplication.
    e1b = numpy.sum(system.H1[0] * G[0]) + numpy.sum(system.H1[1] * G[1])
    if rchol is None:
        rchol = system.rchol_vecs
    nalpha, nbeta = system.nup, system.ndown
    nbasis = system.nbasis
    Ga, Gb = Ghalf[0], Ghalf[1]
    Xa = rchol[0].T.dot(Ga.ravel())
    Xb = rchol[1].T.dot(Gb.ravel())
    ecoul = numpy.dot(Xa, Xa)
    ecoul += numpy.dot(Xb, Xb)
    ecoul += 2 * numpy.dot(Xa, Xb)
    if system.sparse:
        rchol_a, rchol_b = [rchol[0].toarray(), rchol[1].toarray()]
    else:
        rchol_a, rchol_b = rchol[0], rchol[1]

    # T_{abn} = \sum_k Theta_{ak} LL_{ak,n}
    # LL_{ak,n} = \sum_i L_{ik,n} A^*_{ia}

    naux = rchol_a.shape[-1]

    theta = numpy.zeros((naux, nsamples), dtype=numpy.int64)
    for i in range(nsamples):
        theta[:, i] = 2 * numpy.random.randint(0, 2, size=(naux)) - 1

    if control:

        ra = rchol_a.dot(theta).T * numpy.sqrt(1.0 / nsamples)
        rb = rchol_b.dot(theta).T * numpy.sqrt(1.0 / nsamples)

        Ta0 = numpy.zeros((nsamples, nalpha, nalpha), dtype=rchol_a.dtype)
        Tb0 = numpy.zeros((nsamples, nbeta, nbeta), dtype=rchol_b.dtype)

        Ta = numpy.zeros((nsamples, nalpha, nalpha), dtype=rchol_a.dtype)
        Tb = numpy.zeros((nsamples, nbeta, nbeta), dtype=rchol_b.dtype)

        G0aT = C0[:, : system.nup]
        G0bT = C0[:, system.nup :]

        GaT = Ga.T
        GbT = Gb.T

        for x in range(nsamples):
            rmi_a = ra[x].reshape((nalpha, nbasis))
            rmi_b = rb[x].reshape((nbeta, nbasis))

            Ta0[x] = rmi_a.dot(G0aT)
            Tb0[x] = rmi_b.dot(G0bT)
            Ta[x] = rmi_a.dot(GaT)
            Tb[x] = rmi_b.dot(GbT)

        exxa_hf = numpy.tensordot(Ta0, Ta0, axes=((0, 1, 2), (0, 2, 1)))
        exxb_hf = numpy.tensordot(Tb0, Tb0, axes=((0, 1, 2), (0, 2, 1)))

        exxa_corr = numpy.tensordot(Ta, Ta, axes=((0, 1, 2), (0, 2, 1)))
        exxb_corr = numpy.tensordot(Tb, Tb, axes=((0, 1, 2), (0, 2, 1)))

        exxa = exxa0 + (exxa_corr - exxa_hf)
        exxb = exxb0 + (exxb_corr - exxb_hf)

    else:
        rchol_a = rchol_a.reshape((nalpha, nbasis, naux))
        rchol_b = rchol_b.reshape((nbeta, nbasis, naux))

        ra = numpy.einsum("ipX,Xs->ips", rchol_a, theta, optimize=True) * numpy.sqrt(
            1.0 / nsamples
        )
        Gra = numpy.einsum("kq,lqx->lkx", Ga, ra, optimize=True)
        exxa = numpy.tensordot(Gra, Gra, axes=((0, 1, 2), (1, 0, 2)))

        rb = numpy.einsum("ipX,Xs->ips", rchol_b, theta, optimize=True) * numpy.sqrt(
            1.0 / nsamples
        )
        Grb = numpy.einsum("kq,lqx->lkx", Gb, rb, optimize=True)
        exxb = numpy.tensordot(Grb, Grb, axes=((0, 1, 2), (1, 0, 2)))

    exx = exxa + exxb
    e2b = 0.5 * (ecoul - exx)

    # pr.disable()
    # pr.print_stats(sort='tottime')

    return (e1b + e2b + system.ecore, e1b + system.ecore, e2b)


def local_energy_generic_cholesky_opt(
    system, ham, Ga, Gb, Ghalfa, Ghalfb, rchola, rcholb
):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals.

    Parameters
    ----------
    system : :class:`Generic`
        System information for Generic.
    ham : :class:`Abinitio`
        Contains necessary hamiltonian information
    G : :class:`numpy.ndarray`
        Walker's "green's function"
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function" shape is nocc x nbasis
    rchol : :class:`numpy.ndarray`
        trial's half-rotated choleksy vectors

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    # Element wise multiplication.
    if is_cupy(
        rchola
    ):  # if even one array is a cupy array we should assume the rest is done with cupy
        import cupy

        assert cupy.is_available()
        array = cupy.array
        zeros = cupy.zeros
        sum = cupy.sum
        dot = cupy.dot
        trace = cupy.trace
        einsum = cupy.einsum
        isrealobj = cupy.isrealobj
    else:
        array = numpy.array
        zeros = numpy.zeros
        einsum = numpy.einsum
        trace = numpy.trace
        sum = numpy.sum
        dot = numpy.dot
        isrealobj = numpy.isrealobj

    complex128 = numpy.complex128

    e1b = sum(ham.H1[0] * Ga) + sum(ham.H1[1] * Gb)
    nalpha, nbeta = system.nup, system.ndown
    nbasis = ham.nbasis
    if rchola is not None:
        naux = rchola.shape[0]

    if isrealobj(rchola) and isrealobj(rcholb):
        Xa = rchola.dot(Ghalfa.real.ravel()) + 1.0j * rchola.dot(Ghalfa.imag.ravel())
        Xb = rcholb.dot(Ghalfb.real.ravel()) + 1.0j * rcholb.dot(Ghalfb.imag.ravel())
    else:
        Xa = rchola.dot(Ghalfa.ravel())
        Xb = rcholb.dot(Ghalfb.ravel())

    ecoul = dot(Xa, Xa)
    ecoul += dot(Xb, Xb)
    ecoul += 2 * dot(Xa, Xb)

    GhalfaT = Ghalfa.T.copy()  # nbasis x nocc
    GhalfbT = Ghalfb.T.copy()

    Ta = zeros((nalpha, nalpha), dtype=complex128)
    Tb = zeros((nbeta, nbeta), dtype=complex128)

    exx = 0.0j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    if isrealobj(rchola) and isrealobj(rcholb):
        for x in range(naux):  # write a cython function that calls blas for this.
            rmi_a = rchola[x].reshape((nalpha, nbasis))
            rmi_b = rcholb[x].reshape((nbeta, nbasis))
            Ta[:, :].real = rmi_a.dot(GhalfaT.real)
            Ta[:, :].imag = rmi_a.dot(GhalfaT.imag)  # this is a (nalpha, nalpha)
            Tb[:, :].real = rmi_b.dot(GhalfbT.real)
            Tb[:, :].imag = rmi_b.dot(GhalfbT.imag)  # this is (nbeta, nbeta)
            exx += trace(Ta.dot(Ta)) + trace(Tb.dot(Tb))
    else:
        for x in range(naux):  # write a cython function that calls blas for this.
            rmi_a = rchola[x].reshape((nalpha, nbasis))
            rmi_b = rcholb[x].reshape((nbeta, nbasis))
            Ta[:, :] = rmi_a.dot(GhalfaT)  # this is a (nalpha, nalpha)
            Tb[:, :] = rmi_b.dot(GhalfbT)  # this is (nbeta, nbeta)
            exx += trace(Ta.dot(Ta)) + trace(Tb.dot(Tb))

    e2b = 0.5 * (ecoul - exx)

    return (e1b + e2b + ham.ecore, e1b + ham.ecore, e2b)
