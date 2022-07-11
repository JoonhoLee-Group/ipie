import time

import numpy
import scipy.linalg

try:
    from ipie.legacy.estimators.ueg_kernels import (
        build_J_opt, build_K_opt, coulomb_greens_function_per_qvec,
        exchange_greens_function_per_qvec)
except ImportError:
    pass


def exchange_greens_function(nq, kpq_i, kpq, pmq_i, pmq, Gprod, G):
    for iq in range(nq):
        for (idxkpq, i) in zip(kpq[iq], kpq_i[iq]):
            for (idxpmq, j) in zip(pmq[iq], pmq_i[iq]):
                Gprod[iq] += G[j, idxkpq] * G[i, idxpmq]


def coulomb_greens_function(nq, kpq_i, kpq, pmq_i, pmq, Gkpq, Gpmq, G):
    for iq in range(nq):
        for (idxkpq, i) in zip(kpq[iq], kpq_i[iq]):
            Gkpq[iq] += G[i, idxkpq]
        for (idxpmq, i) in zip(pmq[iq], pmq_i[iq]):
            Gpmq[iq] += G[i, idxpmq]


def local_energy_ueg(system, ham, G, Ghalf=None, two_rdm=None):
    """Local energy computation for uniform electron gas
    Parameters
    ----------
    system :
        system class
    ham :
        hamiltonian class
    G :
        Green's function
    Returns
    -------
    etot : float
        total energy
    ke : float
        kinetic energy
    pe : float
        potential energy
    """
    if ham.diagH1:
        ke = numpy.einsum("sii,sii->", ham.H1, G)
    else:
        ke = numpy.einsum("sij,sij->", ham.H1, G)

    Gkpq = numpy.zeros((2, len(ham.qvecs)), dtype=numpy.complex128)
    Gpmq = numpy.zeros((2, len(ham.qvecs)), dtype=numpy.complex128)
    Gprod = numpy.zeros((2, len(ham.qvecs)), dtype=numpy.complex128)

    ne = [system.nup, system.ndown]
    nq = numpy.shape(ham.qvecs)[0]

    for s in [0, 1]:
        # exchange_greens_function(nq, ham.ikpq_i, ham.ikpq_kpq, ham.ipmq_i,ham.ipmq_pmq, Gprod[s],G[s])
        # coulomb_greens_function(nq, ham.ikpq_i, ham.ikpq_kpq,  ham.ipmq_i, ham.ipmq_pmq,Gkpq[s], Gpmq[s],G[s])
        for iq in range(nq):
            Gkpq[s, iq], Gpmq[s, iq] = coulomb_greens_function_per_qvec(
                ham.ikpq_i[iq], ham.ikpq_kpq[iq], ham.ipmq_i[iq], ham.ipmq_pmq[iq], G[s]
            )
            Gprod[s, iq] = exchange_greens_function_per_qvec(
                ham.ikpq_i[iq], ham.ikpq_kpq[iq], ham.ipmq_i[iq], ham.ipmq_pmq[iq], G[s]
            )

    if two_rdm is None:
        two_rdm = numpy.zeros((2, 2, len(ham.qvecs)), dtype=numpy.complex128)
    two_rdm[0, 0] = numpy.multiply(Gkpq[0], Gpmq[0]) - Gprod[0]
    essa = (1.0 / (2.0 * ham.vol)) * ham.vqvec.dot(two_rdm[0, 0])

    two_rdm[1, 1] = numpy.multiply(Gkpq[1], Gpmq[1]) - Gprod[1]
    essb = (1.0 / (2.0 * ham.vol)) * ham.vqvec.dot(two_rdm[1, 1])

    two_rdm[0, 1] = numpy.multiply(Gkpq[0], Gpmq[1])
    two_rdm[1, 0] = numpy.multiply(Gkpq[1], Gpmq[0])
    eos = (1.0 / (2.0 * ham.vol)) * ham.vqvec.dot(two_rdm[0, 1]) + (
        1.0 / (2.0 * ham.vol)
    ) * ham.vqvec.dot(two_rdm[1, 0])

    pe = essa + essb + eos

    return (ke + pe, ke, pe)


# JHLFML
def build_J(system, Gpmq, Gkpq):

    J = [
        numpy.zeros((system.nbasis, system.nbasis), dtype=numpy.complex128),
        numpy.zeros((system.nbasis, system.nbasis), dtype=numpy.complex128),
    ]

    for (iq, q) in enumerate(system.qvecs):
        for idxi, i in enumerate(system.basis):
            for idxj, j in enumerate(system.basis):
                jpq = j + q
                idxjpq = system.lookup_basis(jpq)
                if (idxjpq is not None) and (idxjpq == idxi):
                    J[0][idxj, idxi] += (
                        (1.0 / (2.0 * system.vol))
                        * system.vqvec[iq]
                        * (Gkpq[0][iq] + Gkpq[1][iq])
                    )

    for (iq, q) in enumerate(system.qvecs):
        for idxi, i in enumerate(system.basis):
            for idxj, j in enumerate(system.basis):
                jpq = j - q
                idxjmq = system.lookup_basis(jpq)
                if (idxjmq is not None) and (idxjmq == idxi):
                    J[0][idxj, idxi] += (
                        (1.0 / (2.0 * system.vol))
                        * system.vqvec[iq]
                        * (Gpmq[0][iq] + Gpmq[1][iq])
                    )

    J[1] = J[0]

    return J


def build_K(system, G):
    K = numpy.zeros((2, system.nbasis, system.nbasis), dtype=numpy.complex128)
    for s in [0, 1]:
        for iq in range(len(system.vqvec)):
            for (idxjmq, idxj) in zip(system.ipmq_pmq[iq], system.ipmq_i[iq]):
                for (idxkpq, idxk) in zip(system.ikpq_kpq[iq], system.ikpq_i[iq]):
                    K[s, idxj, idxkpq] += (
                        -(1.0 / (2.0 * system.vol))
                        * system.vqvec[iq]
                        * G[s][idxjmq, idxk]
                    )
        for iq in range(len(system.vqvec)):
            for (idxjpq, idxj) in zip(system.ikpq_kpq[iq], system.ikpq_i[iq]):
                for (idxpmq, idxp) in zip(system.ipmq_pmq[iq], system.ipmq_i[iq]):
                    K[s, idxj, idxpmq] += (
                        -(1.0 / (2.0 * system.vol))
                        * system.vqvec[iq]
                        * G[s][idxjpq, idxp]
                    )
    return K


def fock_ueg(ham, G):
    """Fock matrix computation for uniform electron gas

    Parameters
    ----------
    ham : :class`ipie.legacy.hamiltonians.ueg`
        UEG hamiltonian class.
    G : :class:`numpy.ndarray`
        Green's function.
    Returns
    -------
    F : :class:`numpy.ndarray`
        Fock matrix (2, nbasis, nbasis).
    """
    nbsf = ham.nbasis
    nq = len(ham.qvecs)
    assert nq == len(ham.vqvec)

    Fock = numpy.zeros((2, nbsf, nbsf), dtype=numpy.complex128)
    Gkpq = numpy.zeros((2, nq), dtype=numpy.complex128)
    Gpmq = numpy.zeros((2, nq), dtype=numpy.complex128)

    for s in [0, 1]:
        coulomb_greens_function(
            nq,
            ham.ikpq_i,
            ham.ikpq_kpq,
            ham.ipmq_i,
            ham.ipmq_pmq,
            Gkpq[s],
            Gpmq[s],
            G[s],
        )

    J = build_J_opt(
        nq,
        ham.vqvec,
        ham.vol,
        ham.nbasis,
        ham.ikpq_i,
        ham.ikpq_kpq,
        ham.ipmq_i,
        ham.ipmq_pmq,
        Gkpq,
        Gpmq,
    )

    K = build_K_opt(
        nq,
        ham.vqvec,
        ham.vol,
        ham.nbasis,
        ham.ikpq_i,
        ham.ikpq_kpq,
        ham.ipmq_i,
        ham.ipmq_pmq,
        G,
    )

    for s in [0, 1]:
        Fock[s] = ham.H1[s] + J[s] + K[s]

    return Fock


def unit_test():
    import numpy as np

    from ipie.legacy.systems.ueg import UEG

    inputs = {"nup": 7, "ndown": 7, "rs": 1.0, "ecut": 2.0}
    system = UEG(inputs, True)
    nbsf = system.nbasis
    Pa = np.zeros([nbsf, nbsf], dtype=np.complex128)
    Pb = np.zeros([nbsf, nbsf], dtype=np.complex128)
    na = system.nup
    nb = system.ndown
    for i in range(na):
        Pa[i, i] = 1.0
    for i in range(nb):
        Pb[i, i] = 1.0
    P = np.array([Pa, Pb])
    etot, ekin, epot = local_energy_ueg(system, G=P)
    print("ERHF = {}, {}, {}".format(etot, ekin, epot))

    from ipie.legacy.estimators.greens_function import gab
    from ipie.utils.linalg import exponentiate_matrix, reortho

    # numpy.random.seed()
    rCa = numpy.random.randn(nbsf, na)
    zCa = numpy.random.randn(nbsf, na)
    rCb = numpy.random.randn(nbsf, nb)
    zCb = numpy.random.randn(nbsf, nb)

    Ca = rCa + 1j * zCa
    Cb = rCb + 1j * zCb

    Ca, detR = reortho(Ca)
    Cb, detR = reortho(Cb)
    # S = print(Ca.dot(Cb.T))
    # print(S)
    # exit()
    Ca = numpy.array(Ca, dtype=numpy.complex128)
    Cb = numpy.array(Cb, dtype=numpy.complex128)
    P = [gab(Ca, Ca), gab(Cb, Cb)]
    # diff = P[0] - P[1]
    # print("fro = {}".format(numpy.linalg.norm(diff,ord='fro')))

    # solver = lib.diis.DIIS()

    # dt = 0.1
    # for i in range(100):
    #     # Compute Fock matrix
    #     Fock = fock_ueg(system, G=P)
    #     # Compute DIIS Errvec
    #     PFmFPa = P[0].dot(Fock[0]) - Fock[0].dot(P[0])
    #     PFmFPb = P[1].dot(Fock[1]) - Fock[1].dot(P[1])
    #     errvec = numpy.append(numpy.reshape(PFmFPa, nbsf*nbsf),numpy.reshape(PFmFPb, nbsf*nbsf))
    #     RMS = np.sqrt(np.dot(errvec, errvec))
    #     print ("{} {} {}".format(i,numpy.real(local_energy_ueg(system, P)), numpy.real(RMS)))
    #     # Form Fockvec
    #     Fock[0] = numpy.array(Fock[0])
    #     Fock[1] = numpy.array(Fock[1])
    #     Fockvec = numpy.append(numpy.reshape(Fock[0],nbsf*nbsf), numpy.reshape(Fock[1],nbsf*nbsf))
    #     # Extrapolate Fockvec
    #     # Fockvec = solver.update(Fockvec, xerr=errvec)

    #     # Apply Propagator
    #     Fock = numpy.reshape(Fockvec, (2, nbsf, nbsf))
    #     ea, Ca = numpy.linalg.eig(Fock[0])
    #     eb, Cb = numpy.linalg.eig(Fock[1])
    #     sort_perm = ea.argsort()
    #     ea.sort()
    #     Ca = Ca[:, sort_perm]
    #     sort_perm = eb.argsort()
    #     eb.sort()
    #     Cb = Cb[:, sort_perm]

    #     Ca = Ca[:,:na]
    #     Cb = Cb[:,:nb]
    #     Ca, detR = reortho(Ca)
    #     Cb, detR = reortho(Cb)

    #     P = [gab(Ca, Ca), gab(Cb, Cb)]
    #     # expF = [exponentiate_matrix(-dt*Fock[0]), exponentiate_matrix(-dt*Fock[1])]
    #     # Ca = expF[0].dot(Ca)
    #     # Cb = expF[1].dot(Cb)
    #     # diff = P[0] - P[1]
    #     # print("fro = {}".format(numpy.linalg.norm(diff,ord='fro')))


if __name__ == "__main__":
    unit_test()
