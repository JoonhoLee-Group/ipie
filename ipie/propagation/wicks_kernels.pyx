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

    if nexcit - 1 <= 0:
        for iw in range(nwalker):
            for idet in range(ndet):
                cofactor[iw, idet, 0, 0] = 1.0

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

def get_cofactor_matrix_4_batched(
        int nwalker,
        int ndet,
        int nexcit,
        int row_1,
        int col_1,
        int row_2,
        int col_2,
        double complex[:,:,:,:] det_matrix,
        double complex[:,:,:,:] cofactor):

    cdef int i, j, iw, idet, ishift_1, jshift_1, ishift_2, jshift_2, ncols

    ncols = det_matrix.shape[3] 
    if ncols-2 <=0:
        for iw in range(nwalker):
            for idet in range(ndet):
                cofactor[iw, idet, 0, 0] = 1.0
        return

    for iw in range(nwalker):
        for idet in range(ndet):
            for i in range(nexcit):
                ishift_1 = 0
                jshift_1 = 0
                ishift_2 = 0
                jshift_2 = 0
                if i > row_1:
                    ishift_1 = 1
                if i > row_2:
                    ishift_2 = 1
                if i == nexcit-2 and (row_1 == nexcit-2):
                    continue
                if i == nexcit-1 and (row_2 == nexcit-1):
                    continue
                for j in range(nexcit):
                    if j > col_1:
                        jshift_1 = 1
                    if j > col_2:
                        jshift_2 = 1
                    if j == nexcit-2 and (col_1 == nexcit-2):
                        continue
                    if j == nexcit-1 and (col_2 == nexcit-1):
                        continue
                    # if col_1 == nexcit-2 or col_2 == nexcit-2:
                        # continue
                    # print(i, j, i - (ishift_1+ishift_2), j -
                            # (jshift_1+jshift_2), nexcit-2, row_1, row_2)
                    cofactor[iw, idet, max(i - (ishift_1+ishift_2),0), max(j -
                        (jshift_1+jshift_2),0)] = det_matrix[iw, idet, i, j]
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
