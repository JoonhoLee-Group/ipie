from numba import jit
import numpy

# element wise
@jit(nopython=True, fastmath=True)
def dot_real_cplx(
        A,
        B_real,
        B_cplx,
        ):

    return A * B_real + 1j * (A * B_cplx)


@jit(nopython=True, fastmath=True)
def fill_os_singles(
        cre,
        anh,
        chol_factor,
        spin_buffer,
        det_sls):
    ps = cre[:, 0]
    qs = anh[:, 0]
    ndets = ps.shape[0]
    start = det_sls.start
    for idet in range(ndets):
        spin_buffer[:,start+idet] = chol_factor[:, qs[idet], ps[idet]]

@jit(nopython=True, fastmath=True)
def fill_os_doubles(
        cre,
        anh,
        G0,
        chol_factor,
        spin_buffer,
        det_sls):
    start = det_sls.start
    ndets = cre.shape[0]
    nwalkers = G0.shape[0]
    for iw in range(nwalkers):
        G0_real = G0[iw].real.copy()
        G0_imag = G0[iw].imag.copy()
        for idet in range(ndets):
            p = cre[idet, 0]
            q = anh[idet, 0]
            r = cre[idet, 1]
            s = anh[idet, 1]
            spin_buffer[iw, start + idet, :] = (
                    dot_real_cplx(
                        chol_factor[iw,q,p,:],
                        G0_real[r,s],
                        G0_imag[r,s])
                    -
                    dot_real_cplx(
                        chol_factor[iw,s,p,:],
                        G0_real[r,q],
                        G0_imag[r,q])
                    -
                    dot_real_cplx(
                        chol_factor[iw,q,r,:],
                        G0_real[p,s],
                        G0_imag[p,s])
                    +
                    dot_real_cplx(
                        chol_factor[iw,s,r,:],
                        G0_real[p,q],
                        G0_imag[p,q])
                    )


@jit(nopython=True, fastmath=True)
def fill_os_triples(
        cre,
        anh,
        G0w,
        chol_factor,
        spin_buffer,
        det_sls):
    start = det_sls.start
    ndets = cre.shape[0]
    nwalkers = G0w.shape[0]
    for iw in range(nwalkers):
        G0 = G0w[iw]
        for idet in range(ndets):
            p = cre[idet, 0]
            q = anh[idet, 0]
            r = cre[idet, 1]
            s = anh[idet, 1]
            t = cre[idet, 2]
            u = anh[idet, 2]
            cofac = G0[r,s]*G0[t,u] - G0[r,u]*G0[t,s]
            spin_buffer[iw, start + idet, :] = (
                    dot_real_cplx(
                        chol_factor[iw,q,p],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[r,q]*G0[t,u] - G0[r,u]*G0[t,q]
            spin_buffer[iw, start + idet, :] -= (
                    dot_real_cplx(
                        chol_factor[iw,s,p],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[r,q]*G0[t,s] - G0[r,s]*G0[t,q]
            spin_buffer[iw, start + idet, :] += (
                    dot_real_cplx(
                        chol_factor[iw,u,p],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[p,s]*G0[t,u] - G0[t,s]*G0[p,u]
            spin_buffer[iw, start + idet, :] -= (
                    dot_real_cplx(
                        chol_factor[iw,q,r],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[p,q]*G0[t,u] - G0[t,q]*G0[p,u]
            spin_buffer[iw, start + idet, :] += (
                    dot_real_cplx(
                        chol_factor[iw,s,r],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[p,q]*G0[t,s] - G0[t,q]*G0[p,s]
            spin_buffer[iw, start + idet, :] -= (
                    dot_real_cplx(
                        chol_factor[iw,u,r],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[p,s]*G0[r,u] - G0[r,s]*G0[p,u]
            spin_buffer[iw, start + idet, :] += (
                    dot_real_cplx(
                        chol_factor[iw,q,t],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[p,q]*G0[r,u] - G0[r,q]*G0[p,u]
            spin_buffer[iw, start + idet, :] -= (
                    dot_real_cplx(
                        chol_factor[iw,s,t],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[p,q]*G0[r,s] - G0[r,q]*G0[p,s]
            spin_buffer[iw, start + idet, :] += (
                    dot_real_cplx(
                        chol_factor[iw,u,t],
                        cofac.real,
                        cofac.imag)
                    )

@jit(nopython=True, fastmath=True)
def get_ss_doubles(
        cre,
        anh,
        chol_fact,
        buffer,
        det_sls):
    start = det_sls.start
    ndets = cre.shape[0]
    nwalkers = chol_fact.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = cre[idet, 0]
            q = anh[idet, 0]
            r = cre[idet, 1]
            s = anh[idet, 1]
            buffer[iw, start+idet] += numpy.dot(chol_fact[iw,q,p], chol_fact[iw,s,r]) + 0j
            buffer[iw, start+idet] -= numpy.dot(chol_fact[iw,q,r], chol_fact[iw,s,p]) + 0j

@jit(nopython=True, fastmath=True)
def det_matrix(
        cre,
        anh,
        G0,
        det_matrix):

    nwalker = det_matrix.shape[0]
    ndet = det_matrix.shape[1]
    nex = det_matrix.shape[2]
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

@jit(nopython=True, fastmath=True)
def cofactor_matrix(
        row,
        col,
        det_matrix,
        cofactor):

    nwalker = det_matrix.shape[0]
    ndet = det_matrix.shape[1]
    nexcit = det_matrix.shape[2]
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

@jit(nopython=True, fastmath=True)
def cofactor_matrix_4(
        row_1,
        col_1,
        row_2,
        col_2,
        det_matrix,
        cofactor):

    nwalker = det_matrix.shape[0]
    ndet = det_matrix.shape[1]
    nexcit = det_matrix.shape[2]
    if nexcit - 2 <= 0:
        for iw in range(nwalker):
            for idet in range(ndet):
                cofactor[iw, idet, 0, 0] = 1.0 + 0j
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
                    ii = max(i - (ishift_1+ishift_2),0)
                    jj = max(j - (jshift_1+jshift_2),0)
                    cofactor[iw, idet, ii, jj] = det_matrix[iw, idet, i, j]

@jit(nopython=True, fastmath=True)
def reduce_os_spin_factor(
        ps,
        qs,
        phase,
        cof_mat,
        chol_factor,
        spin_buffer,
        det_sls):
    nwalker = chol_factor.shape[0]
    ndet = cof_mat.shape[1]
    start = det_sls.start
    # assert ndet == det_sls.end - det_sls.start
    for iw in range(nwalker):
        for idet in range(ndet):
            det_cofactor = phase * numpy.linalg.det(cof_mat[iw,idet])
            spin_buffer[iw, start + idet] += dot_real_cplx(
                                            chol_factor[iw, qs[idet], ps[idet]],
                                            det_cofactor.real,
                                            det_cofactor.imag,
                                            )

@jit(nopython=True, fastmath=True)
def fill_os_nfold(
        cre,
        anh,
        det_matrix,
        cof_mat,
        chol_factor,
        spin_buffer,
        det_sls):
    nwalkers = cof_mat.shape[0]
    ndet = cof_mat.shape[1]
    nexcit = det_matrix.shape[-1]
    for iex in range(nexcit):
        ps = cre[:, iex]
        for jex in range(nexcit):
            qs = anh[:, jex]
            cofactor_matrix(
                    iex,
                    jex,
                    det_matrix,
                    cof_mat)
            # nwalkers x ndet
            phase = (-1.0 + 0.j)**(iex + jex)
            reduce_os_spin_factor(
                    ps,
                    qs,
                    phase,
                    cof_mat,
                    chol_factor,
                    spin_buffer,
                    det_sls)

@jit(nopython=True, fastmath=True)
def reduce_ss_spin_factor(
        ps,
        qs,
        rs,
        ss,
        phase,
        cof_mat,
        chol_factor,
        spin_buffer,
        det_sls):
    nwalker = chol_factor.shape[0]
    ndet = cof_mat.shape[1]
    start = det_sls.start
    for iw in range(nwalker):
        for idet in range(ndet):
            det_cofactor = phase * numpy.linalg.det(cof_mat[iw,idet])
            chol_a = chol_factor[iw, ss[idet], rs[idet]]
            chol_b = chol_factor[iw, qs[idet], ps[idet]]
            cont_ab = numpy.dot(chol_a, chol_b)
            spin_buffer[iw, start + idet] += dot_real_cplx(
                                            cont_ab,
                                            det_cofactor.real,
                                            det_cofactor.imag,
                                            )
            chol_c = chol_factor[iw, qs[idet], rs[idet]]
            chol_d = chol_factor[iw, ss[idet], ps[idet]]
            cont_cd = numpy.dot(chol_c, chol_d)
            spin_buffer[iw, start + idet] -= dot_real_cplx(
                                            cont_cd,
                                            det_cofactor.real,
                                            det_cofactor.imag,
                                            )

def get_ss_nfold(
        cre,
        anh,
        dets_mat,
        cof_mat,
        chol_fact,
        buffer,
        det_sls
        ):
    nwalkers = dets_mat.shape[0]
    ndet_level = dets_mat.shape[1]
    nexcit = dets_mat.shape[-1] 
    for iex in range(nexcit):
        for jex in range(nexcit):
            ps = cre[:, iex]
            qs = anh[:, jex]
            for kex in range(iex+1,nexcit):
                rs = cre[:, kex]
                for lex in range(jex+1,nexcit):
                    ss = anh[:, lex]
                    cofactor_matrix_4(
                                    iex,
                                    jex,
                                    kex,
                                    lex,
                                    dets_mat,
                                    cof_mat)
                    phase = (-1.0 + 0.0j)**(kex + lex + iex + jex)
                    reduce_ss_spin_factor(
                            ps,
                            qs,
                            rs,
                            ss,
                            phase,
                            cof_mat,
                            chol_fact,
                            buffer,
                            det_sls)
