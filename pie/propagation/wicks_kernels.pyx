import numpy
cimport numpy

numpy.import_array()

def get_det_matrix_batched(
        int nex,
        numpy.ndarray cre,
        numpy.ndarray anh,
        numpy.ndarray G0,
        double complex[:,:,:,:] det_matrix):

    cdef int iw, idet, iex, jex
    cdef int nwalker = G0.shape[0]
    cdef int ndet = cre.shape[0]
    cdef int p, q, r, s

    for iw in range(nwalker):
        for idet in range(ndet):
            for iex in range(nex):
                p = cre[idet][iex]
                q = anh[idet][iex]
                det_matrix[iw, idet, iex, iex] = G0[iw, p, q]
                for jex in range(iex+1, nex):
                    r = cre[idet][jex]
                    s = anh[idet][jex]
                    det_matrix[iw, idet, iex, jex] = G0[iw, p, s]
                    det_matrix[iw, idet, jex, iex] = G0[iw, r, q]
    return det_matrix

def reduce_to_CI_tensor(
        int nwalker,
        int ndet_level,
        int[:] ps,
        int[:] qs,
        double complex[:,:,:] tensor,
        double complex[:,:] rhs):
    cdef iw, idet
    for iw in range(nwalker):
        for idet in range(ndet_level):
            # += not supported in cython for complex types.
            tensor[iw, ps[idet], qs[idet]] = tensor[iw, ps[idet], qs[idet]] + rhs[iw, idet]

def get_cofactor_matrix_batched(
        int nwalker,
        int ndet,
        int nexcit,
        int row,
        int col,
        double complex[:,:,:,:] det_matrix,
        double complex[:,:,:,:] cofactor):
    cdef int i, j, iw, idet, ishift, jshift

    for iw in range(nwalker):
        for idet in range(ndet):
            for i in range(nexcit):
                ishift = 0
                jshift = 0
                if i > row:
                    ishift = 1
                if i == nexcit-1 and row == nexcit - 1:
                    continue
                for j in range(nexcit):
                    if j > col:
                        jshift = 1
                    if j == nexcit-1 and col == nexcit - 1:
                        continue
                    cofactor[iw, idet, i - ishift, j - jshift] = det_matrix[iw, idet, i, j]

# def get_greens_function_det_matrix_batched(
            # int nex,
            # numpy.ndarray cre,
            # numpy.ndarray anh,
            # numpy.ndarray G0,
            # double complex[:,:,:,:] cofactor,
            # int[:,:] signs):

    # cdef int iw, idet, iex, jex
    # cdef int nwalker = G0.shape[0]
    # cdef int ndet = cre.shape[0]
    # cdef np.ndarray det_matrix = np.zeros([nex, nex], dtype=np.complex128)
    # for iw in range(nwalker):
        # for idet in range(ndet):
            # # need to batch over p_q, do this in phases.
            # for iex in range(nex_a):
                # cdef int p = cre[jdet][iex]
                # for jex in range(nex_a):
                    # cdef int q = anh[jdet][jex]
                    # cofactor[iw, idet, ] = get_cofactor(det_a, iex, jex)
                    # CIa[iw,q,p] += phase * (-1)**(iex+jex) * numpy.linalg.det(cofactor)
