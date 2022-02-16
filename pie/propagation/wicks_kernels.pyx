import numpy
cimport numpy
import math

def get_det_matrix_batched(
        int nex,
        numpy.ndarray cre,
        numpy.ndarray anh,
        numpy.ndarray G0,
        double complex[:,:,:,:] det_matrix):

    cdef int iw, idet, iex, jex
    cdef int nwalker = G0.shape[0]
    cdef int ndet = cre.shape[0]

    for iw in range(nwalker):
        for idet in range(ndet):
            for iex in range(nex):
                p = cre[idet][iex]
                q = anh[idet][iex]
                det_matrix[iw,idet,iex,iex] = G0[iw,p,q]
                for jex in range(iex+1, nex):
                    r = cre[idet][jex]
                    s = anh[idet][jex]
                    det_matrix[iw, idet, iex, jex] = G0[iw, p, s]
                    det_matrix[iw, idet, jex, iex] = G0[iw, r, q]
    return det_matrix
