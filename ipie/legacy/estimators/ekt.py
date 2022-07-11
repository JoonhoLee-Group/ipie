import numpy

try:
    from pyscf import lib

    einsum = lib.einsum
except (ImportError, ValueError, OSError):
    einsum = numpy.einsum

# Assume it's generic
# 4-fold integral symmetry
def ekt_1p_fock_opt(h1, cholvec, rdm1a, rdm1b):
    nmo = rdm1a.shape[0]
    assert len(cholvec.shape) == 3
    assert cholvec.shape[1] * cholvec.shape[2] == nmo * nmo
    nchol = cholvec.shape[0]

    I = numpy.eye(nmo)
    gamma = I - rdm1a.T + I - rdm1b.T
    rdm1 = rdm1a + rdm1b

    Xa = cholvec.reshape((nchol, nmo * nmo)).dot(rdm1a.ravel())
    Xb = cholvec.reshape((nchol, nmo * nmo)).dot(rdm1b.ravel())

    Xachol = numpy.tensordot(Xa, cholvec.transpose(0, 2, 1), axes=([0], [0]))
    Xbchol = numpy.tensordot(Xb, cholvec.transpose(0, 2, 1), axes=([0], [0]))

    J = (
        2.0 * (Xachol + Xbchol)
        - 2.0 * rdm1a.T.dot(Xbchol)
        - rdm1a.T.dot(Xachol)
        - rdm1b.T.dot(Xbchol)
    )

    K = numpy.zeros_like(J)

    for x in range(nchol):
        c = cholvec[x, :, :]
        c2 = cholvec[x, :, :].T
        K += -c.dot(rdm1.T).dot(c2)
        K += rdm1a.T.dot(c).dot(rdm1a.T).dot(c2)
        K += rdm1b.T.dot(c).dot(rdm1b.T).dot(c2)

    Fock = gamma.dot(h1) + J + K

    cholvec = cholvec.T.reshape((nmo * nmo, nchol))

    return Fock


# Assume it's generic
# 4-fold integral symmetry
def ekt_1h_fock_opt(h1, cholvec, rdm1a, rdm1b):
    nmo = rdm1a.shape[0]
    assert len(cholvec.shape) == 3
    assert cholvec.shape[1] * cholvec.shape[2] == nmo * nmo
    nchol = cholvec.shape[0]

    Xa = cholvec.reshape((nchol, nmo * nmo)).dot(rdm1a.ravel())
    Xb = cholvec.reshape((nchol, nmo * nmo)).dot(rdm1b.ravel())

    # X[n] = \sum_ik L[n,ik] G[ik]
    # Xchol[i,k] = \sum_n X[n] L[n,ik]
    Xachol = numpy.tensordot(Xa, cholvec.transpose(0, 2, 1), axes=([0], [0]))
    Xbchol = numpy.tensordot(Xb, cholvec.transpose(0, 2, 1), axes=([0], [0]))

    # J[i,j] = -\sum_s \sum_k Gs[i,k] Xschol[j,k]
    J = -2.0 * rdm1a.dot(Xbchol.T) - rdm1a.dot(Xachol.T) - rdm1b.dot(Xbchol.T)

    K = numpy.zeros_like(J)

    for x in range(nchol):
        c = cholvec[x, :, :]
        c2 = cholvec[x, :, :].T
        K += rdm1a.dot(c.T).dot(rdm1a).dot(c2.T)
        K += rdm1a.dot(c.T).dot(rdm1b).dot(c2.T)

    gamma = rdm1a + rdm1b
    Fock = -gamma.dot(h1.T) + J + K

    return Fock


def gen_fock_1h(h1, cholvec, rdm1a, rdm1b):
    """Generalised Fock matrix for 1-hole excitation.

    F[s,p,q] = < c_{ps}^+ {H,c_{qs}} >
    """
    nmo = rdm1a.shape[0]
    assert len(cholvec.shape) == 3
    assert cholvec.shape[1] * cholvec.shape[2] == nmo * nmo
    nchol = cholvec.shape[0]

    # Coulomb:
    # s,t = spin variables
    # p,q,i,k,l = basis functions
    # J[s,p,q] = -\sum_{ikl,t} L[n,i,k] L[n,q,l] P[s,p,l] P[t,i,k]
    # 1. Form X[t,n] = \sum_{ik} L[n,i,k] P[t,i,k]
    Xa = cholvec.reshape((nchol, nmo * nmo)).dot(rdm1a.ravel())
    Xb = cholvec.reshape((nchol, nmo * nmo)).dot(rdm1b.ravel())
    Ta = numpy.einsum("nql,pl->pqn", cholvec, rdm1a)
    # for n in range(nchol):
    # for i in range(nmo):
    # for j in range(nmo):
    # if abs(Ta[i,j,n]) > 1e-6:
    # print("{} {} {} {}".format(n,i,j,Ta[i,j,n].real))
    # 2. Form Xchol[q,l] = \sum_n X[n] L[n,q,l]
    Xachol = numpy.tensordot(Xa, cholvec, axes=([0], [0]))
    Xbchol = numpy.tensordot(Xb, cholvec, axes=([0], [0]))
    # 3. J[s,p,q] = -\sum_l P[s,p,l] Xchol[q,l]
    Ja = -numpy.dot(rdm1a, Xachol.T)
    Ja += -numpy.dot(rdm1a, Xbchol.T)
    Jb = -numpy.dot(rdm1b, Xachol.T)
    Jb += -numpy.dot(rdm1b, Xbchol.T)
    # JJ = -2*numpy.einsum('pqn,n->pq', Ta, Xa)
    # Ja = JJ
    # print(JJ[0,0], Ja[0,0])
    # print("DELTA: ", numpy.linalg.norm(JJ-Ja))

    # Exchange:
    # K[s,p,q] = \sum_{ikl} L[n,i,k] L[n,q,l] P[s,p,k] P[s,i,l]
    # 1. T[s,n,i,p] = \sum_k L[n,i,k] P[s,p,k]
    Ta = numpy.tensordot(cholvec, rdm1a, axes=([2], [1]))
    Tb = numpy.tensordot(cholvec, rdm1b, axes=([2], [1]))
    # K[p,q] = \sum_{n,i} T[n,i,p] T[n,q,i]
    Ka = numpy.tensordot(Ta, Ta, axes=([0, 1], [0, 2]))
    Kb = numpy.tensordot(Tb, Tb, axes=([0, 1], [0, 2]))

    # One-Body:
    F1a = -numpy.dot(rdm1a, h1.T)
    F1b = -numpy.dot(rdm1b, h1.T)

    Focka = F1a + Ja + Ka
    Fockb = F1b + Jb + Kb

    return (Focka, Fockb)


def gen_fock_1p(h1, cholvec, rdm1a, rdm1b):
    """Generalised Fock matrix for 1-particle excitation.

    F[s,p,q] = < c_{ps} {H,c_{qs}^+} >
    """
    nmo = rdm1a.shape[0]
    assert len(cholvec.shape) == 3
    assert cholvec.shape[1] * cholvec.shape[2] == nmo * nmo
    nchol = cholvec.shape[0]

    # Coulomb:
    # s,t = spin variables
    # p,q,i,k,l = basis functions
    # J[s,p,q] = -\sum_{ijl,t} L[n,i,q] L[n,j,l] P[s,i,p] P[t,j,l]
    # 1. Form X[t,n] = \sum_{jl} L[n,j,l] P[t,j,l]
    Xa = cholvec.reshape((nchol, nmo * nmo)).dot(rdm1a.ravel())
    Xb = cholvec.reshape((nchol, nmo * nmo)).dot(rdm1b.ravel())
    # 2. Form Xchol[i,q] = \sum_n X[n] L[n,i,q]
    Xachol = numpy.tensordot(Xa, cholvec, axes=([0], [0]))
    Xbchol = numpy.tensordot(Xb, cholvec, axes=([0], [0]))

    # Coulomb
    # J[s,p,q] = \sum_i P[s,i,p] Xchol[i,q]
    Ja = -numpy.tensordot(rdm1a, Xachol, axes=([0], [0]))
    Ja += -numpy.tensordot(rdm1a, Xbchol, axes=([0], [0]))
    Jb = -numpy.tensordot(rdm1b, Xbchol, axes=([0], [0]))
    Jb += -numpy.tensordot(rdm1b, Xachol, axes=([0], [0]))
    # Additional Coulomb like term:
    # J1b = \sum_{jlt} L[n,p,q] L[n,j,l] P[t,j,l]
    #     = \sum_{t} X[t,n] L[n,p,q]
    Ja += Xachol + Xbchol
    Jb += Xachol + Xbchol

    # Exchange
    # K[s,p,q] = \sum_{ikl} L[n,i,q] L[n,j,l] P[s,j,p] P[s,i,l]
    # 1. T[s,n,l,p] = \sum_j L[n,j,l] P[s,j,p]
    Ta = numpy.tensordot(cholvec, rdm1a, axes=([1], [0]))
    Tb = numpy.tensordot(cholvec, rdm1b, axes=([1], [0]))
    # 2. K[s,p,q] = \sum_{nl} T[s,n,l,p] T[s,n,q,l]
    Ka = numpy.tensordot(Ta, Ta, axes=([0, 1], [0, 2]))
    Kb = numpy.tensordot(Tb, Tb, axes=([0, 1], [0, 2]))
    # Additional Exchange like term:
    # K1b[s,p,q] = -\sum_{jl} L[n,j,q] L[n,p,l] P[s,j,l]
    #            = -\sum_{t} L[n,p,l] T[s,n,q,l]
    Ka -= numpy.tensordot(cholvec, Ta, axes=([0, 2], [0, 2]))
    Kb -= numpy.tensordot(cholvec, Tb, axes=([0, 2], [0, 2]))

    # 'One-body' bits
    # F1[p,q] = \sum_i h[i,q] (delta[i,p] - P[s,i,p])
    I = numpy.eye(nmo)
    F1a = numpy.dot((I - rdm1a).T, h1)
    F1b = numpy.dot((I - rdm1b).T, h1)

    Focka = F1a + Ja + Ka
    Fockb = F1b + Jb + Kb

    return (Focka, Fockb)


# import numpy
# try:
#     from pyscf import lib
#     einsum = lib.einsum
# except ImportError:
#     einsum = numpy.einsum

# def ekt_1p_fock(h1, cholvec, rdm1a, rdm1b):

#     nmo = rdm1a.shape[0]

#     assert (len(cholvec.shape) == 3)

#     nchol = cholvec.shape[0]
#     assert(nmo == cholvec.shape[1])
#     assert(nmo == cholvec.shape[2])

#     I = numpy.eye(nmo)

#     gamma = I - rdm1a.T + I - rdm1b.T
#     rdm1 = rdm1a + rdm1b

#     Fock = einsum("rq,pr->pq",h1, gamma)\
#     + 2.0 * einsum("xpq,xrs,rs->pq", cholvec, cholvec, rdm1) - einsum("xps,xrq,rs->pq", cholvec, cholvec, rdm1)\
#     - 2.0 * einsum("rp,ts,xrq,xts->pq", rdm1a, rdm1b, cholvec, cholvec)\
#     + einsum("rs,tp,xrq,xts->pq",rdm1a, rdm1a, cholvec, cholvec)\
#     - einsum("rp,ts,xrq,xts->pq",rdm1a, rdm1a, cholvec, cholvec)\
#     + einsum("rs,tp,xrq,xts->pq",rdm1b, rdm1b, cholvec, cholvec)\
#     - einsum("rp,ts,xrq,xts->pq",rdm1b, rdm1b, cholvec, cholvec)

#     return Fock

# def ekt_1h_fock(h1, cholvec, rdm1a, rdm1b):

#     nmo = rdm1a.shape[0]

#     assert (len(cholvec.shape) == 3)
#     nchol = cholvec.shape[0]
#     assert(nmo == cholvec.shape[1])
#     assert(nmo == cholvec.shape[2])

#     Fock = -einsum("qr,pr->pq",h1, (rdm1a+rdm1b))\
#     - 2.0 * einsum("xqr,xts,pr,ts->pq",cholvec, cholvec, rdm1a, rdm1b)\
#     + einsum("xtr,xqs,pr,ts->pq",cholvec, cholvec, rdm1a, rdm1a)\
#     - einsum("xtr,xqs,ps,tr->pq",cholvec, cholvec, rdm1a, rdm1a)\
#     + einsum("xtr,xqs,pr,ts->pq",cholvec, cholvec, rdm1b, rdm1b)\
#     - einsum("xtr,xqs,ps,tr->pq",cholvec, cholvec, rdm1b, rdm1b)

#     return Fock

# def ekt_1p_fock_opt(h1, cholvec, rdm1a, rdm1b):

#     nmo = rdm1a.shape[0]

#     assert (len(cholvec.shape) == 3)

#     nchol = cholvec.shape[0]
#     assert(nmo == cholvec.shape[1])
#     assert(nmo == cholvec.shape[2])

#     I = numpy.eye(nmo)

#     gamma = I - rdm1a.T + I - rdm1b.T
#     rdm1 = rdm1a + rdm1b

#     Xa = cholvec.reshape((nchol, nmo*nmo)).dot(rdm1a.ravel())
#     Xb = cholvec.reshape((nchol, nmo*nmo)).dot(rdm1b.ravel())

#     Xachol = numpy.tensordot(Xa, cholvec, axes=([0],[0]))
#     Xbchol = numpy.tensordot(Xb, cholvec, axes=([0],[0]))

#     J = 2.0 * (Xachol + Xbchol) - 2.0 * rdm1a.T.dot(Xbchol) - rdm1a.T.dot(Xachol)\
#     - rdm1b.T.dot(Xbchol)

#     K = numpy.zeros_like(J)

#     for x in range(nchol):
#         c = cholvec[x,:,:]
#         K += - c.dot(rdm1.T).dot(c)
#         K += rdm1a.T.dot(c).dot(rdm1a.T).dot(c)
#         K += rdm1b.T.dot(c).dot(rdm1b.T).dot(c)

#     Fock = gamma.dot(h1) + J + K

#     return Fock

# def ekt_1h_fock_opt(h1, cholvec, rdm1a, rdm1b):
#     nmo = rdm1a.shape[0]

#     assert (len(cholvec.shape) == 3)
#     nchol = cholvec.shape[0]
#     assert(nmo == cholvec.shape[1])
#     assert(nmo == cholvec.shape[2])

#     Xa = cholvec.reshape((nchol, nmo*nmo)).dot(rdm1a.ravel())
#     Xb = cholvec.reshape((nchol, nmo*nmo)).dot(rdm1b.ravel())

#     Xachol = numpy.tensordot(Xa, cholvec, axes=([0],[0]))
#     Xbchol = numpy.tensordot(Xb, cholvec, axes=([0],[0]))

#     J = - 2.0 * rdm1a.dot(Xbchol.T) - rdm1a.dot(Xachol.T) - rdm1b.dot(Xbchol.T)

#     K = numpy.zeros_like(J)

#     for x in range(nchol):
#         c = cholvec[x,:,:]
#         K += rdm1a.dot(c.T).dot(rdm1a).dot(c.T)
#         K += rdm1a.dot(c.T).dot(rdm1b).dot(c.T)

#     gamma = rdm1a+rdm1b
#     Fock = - gamma.dot(h1.T) + J + K

#     return Fock
