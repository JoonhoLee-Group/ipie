import numpy
cimport numpy

numpy.import_array()

def pack_cholesky(
        long[:] idx_i,
        long[:] idx_j,
        double[:,:] Lchol_packed,
        double[:,:,:] Lchol):

    cdef long i, x
    cdef long nbsf, nut, nchol
    nchol = Lchol.shape[2]
    nbsf = Lchol.shape[0]
    nut = nbsf *(nbsf+1)//2

    for i in range(nut):
        for x in range(nchol):
            Lchol_packed[i,x] = Lchol[idx_i[i],idx_j[i],x]
    return

def unpack_VHS(
        long[:] idx_i,
        long[:] idx_j,
        double complex[:] VHS_packed,
        double complex[:,:] VHS):

    cdef long i, j
    cdef long nbsf, nut
    nbsf = VHS.shape[0]
    nut = round(nbsf *(nbsf+1)/2)

    for i in range(nut):
        VHS[idx_i[i],idx_j[i]] = VHS_packed[i]
        VHS[idx_j[i],idx_i[i]] = VHS_packed[i]

    return

def unpack_VHS_sp(
        long[:] idx_i,
        long[:] idx_j,
        float complex[:] VHS_packed,
        float complex[:,:] VHS):

    cdef long i, j
    cdef long nbsf, nut
    nbsf = VHS.shape[0]
    nut = round(nbsf *(nbsf+1)/2)

    for i in range(nut):
        VHS[idx_i[i],idx_j[i]] = VHS_packed[i]
        VHS[idx_j[i],idx_i[i]] = VHS_packed[i]

    return


def unpack_VHS_batch(
        long[:] idx_i,
        long[:] idx_j,
        double complex[:,:] VHS_packed,
        double complex[:,:,:] VHS):

    cdef long i, iw
    cdef long nbsf, nut, nwalkers
    nwalkers = VHS.shape[0]
    nbsf = VHS.shape[1]
    nut = round(nbsf *(nbsf+1)/2)

    for iw in range(nwalkers):
        for i in range(nut):
            VHS[iw, idx_i[i],idx_j[i]] = VHS_packed[iw,i]
            VHS[iw, idx_j[i],idx_i[i]] = VHS_packed[iw,i]

    return

def unpack_VHS_batch_sp(
        long[:] idx_i,
        long[:] idx_j,
        float complex[:,:] VHS_packed,
        float complex[:,:,:] VHS):

    cdef long i, iw
    cdef long nbsf, nut, nwalkers
    nwalkers = VHS.shape[0]
    nbsf = VHS.shape[1]
    nut = round(nbsf *(nbsf+1)/2)

    for iw in range(nwalkers):
        for i in range(nut):
            VHS[iw, idx_i[i],idx_j[i]] = VHS_packed[iw,i]
            VHS[iw, idx_j[i],idx_i[i]] = VHS_packed[iw,i]

    return

# assume chol is fast
def pack_cholesky_fast(
        long[:] idx_i,
        long[:] idx_j,
        double[:,:] Lchol_packed,
        double[:,:,:] Lchol):

    cdef long i, j, x
    cdef long nbsf, nut, nchol

    nchol = Lchol.shape[0]
    nbsf = Lchol.shape[1]
    nut = round(nbsf *(nbsf+1)/2)

    for x in range(nchol):
        for i in range(nut):
            Lchol_packed[x,i] = Lchol[x,idx_i[i],idx_j[i]]

    return
