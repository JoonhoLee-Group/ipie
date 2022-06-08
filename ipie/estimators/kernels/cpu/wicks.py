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


# Overlap
@jit(nopython=True, fastmath=True)
def get_dets_singles(
        cre,
        anh,
        offset,
        G0,
        dets
        ):
    ps = cre[:, 0] + offset
    qs = anh[:, 0] + offset
    ndets = ps.shape[0]
    nwalkers = G0.shape[0]
    for idet in range(ndets):
        dets[:,idet] = G0[:, ps[idet], qs[idet]]

@jit(nopython=True, fastmath=True)
def get_dets_doubles(
        cre,
        anh,
        offset,
        G0,
        dets
        ):
    ps = cre[:, 0] + offset
    qs = anh[:, 0] + offset
    rs = cre[:, 1] + offset
    ss = anh[:, 1] + offset
    ndets = ps.shape[0]
    nwalkers = G0.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            dets[iw,idet] = (
                    G0[iw, ps[idet], qs[idet]] * G0[iw, rs[idet], ss[idet]]
                    -
                    G0[iw, ps[idet], ss[idet]] * G0[iw, rs[idet], qs[idet]]
                    )

@jit(nopython=True, fastmath=True)
def get_dets_triples(
        cre,
        anh,
        offset,
        G0,
        dets,
        ):
    ndets = len(cre)
    nwalkers = G0.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            ps, qs = cre[idet,0] + offset, anh[idet,0] + offset
            rs, ss = cre[idet,1] + offset, anh[idet,1] + offset
            ts, us = cre[idet,2] + offset, anh[idet,2] + offset
            dets[iw,idet] = G0[iw, ps, qs]*(
                        G0[iw, rs, ss]*G0[iw, ts, us] - G0[iw, rs, us]*G0[iw, ts, ss]
                         )
            dets[iw,idet] -= G0[iw, ps, ss]*(
                        G0[iw, rs, qs]*G0[iw, ts, us] - G0[iw, rs, us]*G0[iw, ts, qs]
                        )
            dets[iw,idet] += G0[iw, ps, us]*(
                        G0[iw, rs, qs]*G0[iw, ts, ss] - G0[iw, rs, ss]*G0[iw, ts, qs]
                        )

@jit(nopython=True, fastmath=True)
def get_dets_nfold(
        cre,
        anh,
        offset,
        G0,
        dets
        ):
    ndets = len(cre)
    nwalkers = G0.shape[0]
    nex = cre.shape[-1]
    det = numpy.zeros((nex, nex), dtype=numpy.complex128)
    for iw in range(nwalkers):
        for idet in range(ndets):
            for iex in range(nex):
                p = cre[idet, iex] + offset
                q = anh[idet, iex] + offset
                det[iex,iex] = G0[iw,p,q]
                for jex in range(iex+1, nex):
                    r = cre[idet, jex] + offset
                    s = anh[idet, jex] + offset
                    det[iex, jex] = G0[iw,p,s]
                    det[jex, iex] = G0[iw,r,q]
            dets[iw, idet] = numpy.linalg.det(det)

@jit(nopython=True, fastmath=True)
def build_det_matrix(
        cre,
        anh,
        offset,
        G0,
        det_mat):

    nwalker = det_mat.shape[0]
    ndet = det_mat.shape[1]
    if ndet == 0:
        return
    nex = det_mat.shape[2]
    for iw in range(nwalker):
        for idet in range(ndet):
            for iex in range(nex):
                p = cre[idet,iex] + offset
                q = anh[idet,iex] + offset
                det_mat[iw, idet, iex, iex] = G0[iw, p, q]
                for jex in range(iex+1, nex):
                    r = cre[idet,jex] + offset
                    s = anh[idet,jex] + offset
                    det_mat[iw, idet, iex, jex] = G0[iw, p, s]
                    det_mat[iw, idet, jex, iex] = G0[iw, r, q]


# Green's function

@jit(nopython=False, fastmath=False)
def reduce_CI_singles(
        cre,
        anh,
        mapping,
        phases,
        CI):
    ps = cre[:,0]
    qs = anh[:,0]
    ndets = len(cre)
    nwalkers = phases.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[ps[idet]]
            q = qs[idet]
            CI[iw, q, p] += phases[iw, idet]

@jit(nopython=True, fastmath=True)
def reduce_CI_doubles(
        cre,
        anh,
        mapping,
        offset,
        phases,
        G0,
        CI):
    ps = cre[:,0]
    qs = anh[:,0]
    rs = cre[:, 1]
    ss = anh[:, 1]
    ndets = len(cre)
    nwalkers = G0.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[ps[idet]]
            q = qs[idet]
            r = mapping[rs[idet]]
            s = ss[idet]
            po = ps[idet] + offset
            qo = qs[idet] + offset
            ro = rs[idet] + offset
            so = ss[idet] + offset
            CI[iw, q, p] += phases[iw, idet] * G0[iw,ro,so]
            CI[iw, s, r] += phases[iw, idet] * G0[iw,po,qo]
            CI[iw, q, r] -= phases[iw, idet] * G0[iw,po,so]
            CI[iw, s, p] -= phases[iw, idet] * G0[iw,ro,qo]

@jit(nopython=True, fastmath=True)
def reduce_CI_triples(
        cre,
        anh,
        mapping,
        offset,
        phases,
        G0,
        CI):
    ps = cre[:, 0]
    qs = anh[:, 0]
    rs = cre[:, 1]
    ss = anh[:, 1]
    ts = cre[:, 2]
    us = anh[:, 2]
    ndets = len(cre)
    nwalkers = G0.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[ps[idet]]
            q = qs[idet]
            r = mapping[rs[idet]]
            s = ss[idet]
            t = mapping[ts[idet]]
            u = us[idet]
            po = ps[idet] + offset
            qo = qs[idet] + offset
            ro = rs[idet] + offset
            so = ss[idet] + offset
            to = ts[idet] + offset
            uo = us[idet] + offset
            CI[iw,q,p] += phases[iw,idet] * (
                    G0[iw,ro,so]*G0[iw,to,uo] -
                    G0[iw,ro,uo]*G0[iw,to,so]
                    ) # 0 0
            CI[iw,s,p] -= phases[iw,idet] * (
                    G0[iw,ro,qo]*G0[iw,to,uo] -
                    G0[iw,ro,uo]*G0[iw,to,qo]
                    ) # 0 1
            CI[iw,u,p] += phases[iw,idet] * (
                    G0[iw,ro,qo]*G0[iw,to,so] -
                    G0[iw,ro,so]*G0[iw,to,qo]
                    ) # 0 2

            CI[iw,q,r] -= phases[iw,idet] * (
                    G0[iw,po,so]*G0[iw,to,uo] -
                    G0[iw,po,uo]*G0[iw,to,so]
                    ) # 1 0
            CI[iw,s,r] += phases[iw,idet] * (
                    G0[iw,po,qo]*G0[iw,to,uo] -
                    G0[iw,po,uo]*G0[iw,to,qo]
                    ) # 1 1
            CI[iw,u,r] -= phases[iw,idet] * (
                    G0[iw,po,qo]*G0[iw,to,so] -
                    G0[iw,po,so]*G0[iw,to,qo]
                    ) # 1 2

            CI[iw,q,t] += phases[iw,idet] * (
                    G0[iw,po,so]*G0[iw,ro,uo] -
                    G0[iw,po,uo]*G0[iw,ro,so]
                    ) # 2 0
            CI[iw,s,t] -= phases[iw,idet] * (
                    G0[iw,po,qo]*G0[iw,ro,uo] -
                    G0[iw,po,uo]*G0[iw,ro,qo]
                    ) # 2 1
            CI[iw,u,t] += phases[iw,idet] * (
                    G0[iw,po,qo]*G0[iw,ro,so] -
                    G0[iw,po,so]*G0[iw,ro,qo]
                    ) # 2 2


@jit(nopython=True, fastmath=True)
def _reduce_nfold_cofactor_contribution(
        ps,
        qs,
        mapping,
        sign,
        phases,
        cofactor_matrix,
        CI
        ):
    nwalkers = cofactor_matrix.shape[0]
    ndets = cofactor_matrix.shape[1]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[ps[idet]]
            q = qs[idet]
            det = numpy.linalg.det(cofactor_matrix[iw, idet])
            rhs = sign  * det * phases[iw, idet]
            CI[iw, q, p] += rhs

@jit(nopython=True, fastmath=True)
def reduce_CI_nfold(
        cre,
        anh,
        mapping,
        offset,
        phases,
        det_mat,
        cof_mat,
        CI):
    ndets = len(cre)
    nwalkers = CI.shape[0]
    nexcit = det_mat.shape[-1]
    for iex in range(nexcit):
        p = cre[:, iex]
        for jex in range(nexcit):
            q = anh[:, jex]
            build_cofactor_matrix(
                    iex,
                    jex,
                    det_mat,
                    cof_mat)
            sign = (-1 + 0.0j)**(iex + jex)
            _reduce_nfold_cofactor_contribution(
                    p,
                    q,
                    mapping,
                    sign,
                    phases,
                    cof_mat,
                    CI
                    )


# Energy evaluation
@jit(nopython=True, fastmath=True)
def fill_os_singles(
        cre,
        anh,
        mapping,
        offset,
        chol_factor,
        spin_buffer,
        det_sls):
    ps = cre[:, 0]
    qs = anh[:, 0]
    ndets = ps.shape[0]
    start = det_sls.start
    for idet in range(ndets):
        spin_buffer[:,start+idet] = chol_factor[:, qs[idet], mapping[ps[idet]]]

@jit(nopython=True, fastmath=True)
def fill_os_doubles(
        cre,
        anh,
        mapping,
        offset,
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
            p = mapping[cre[idet, 0]]
            q = anh[idet, 0]
            r = mapping[cre[idet, 1]]
            s = anh[idet, 1]
            po = cre[idet, 0] + offset
            qo = anh[idet, 0] + offset
            ro = cre[idet, 1] + offset
            so = anh[idet, 1] + offset
            spin_buffer[iw, start + idet, :] = (
                    dot_real_cplx(
                        chol_factor[iw,q,p,:],
                        G0_real[ro,so],
                        G0_imag[ro,so])
                    -
                    dot_real_cplx(
                        chol_factor[iw,s,p,:],
                        G0_real[ro,qo],
                        G0_imag[ro,qo])
                    -
                    dot_real_cplx(
                        chol_factor[iw,q,r,:],
                        G0_real[po,so],
                        G0_imag[po,so])
                    +
                    dot_real_cplx(
                        chol_factor[iw,s,r,:],
                        G0_real[po,qo],
                        G0_imag[po,qo])
                    )


@jit(nopython=True, fastmath=True)
def fill_os_triples(
        cre,
        anh,
        mapping,
        offset,
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
            p = mapping[cre[idet, 0]]
            q = anh[idet, 0]
            r = mapping[cre[idet, 1]]
            s = anh[idet, 1]
            t = mapping[cre[idet, 2]]
            u = anh[idet, 2]
            po = cre[idet, 0] + offset
            qo = anh[idet, 0] + offset
            ro = cre[idet, 1] + offset
            so = anh[idet, 1] + offset
            to = cre[idet, 2] + offset
            uo = anh[idet, 2] + offset
            cofac = G0[ro,so]*G0[to,uo] - G0[ro,uo]*G0[to,so]
            spin_buffer[iw, start + idet, :] = (
                    dot_real_cplx(
                        chol_factor[iw,q,p],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[ro,qo]*G0[to,uo] - G0[ro,uo]*G0[to,qo]
            spin_buffer[iw, start + idet, :] -= (
                    dot_real_cplx(
                        chol_factor[iw,s,p],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[ro,qo]*G0[to,so] - G0[ro,so]*G0[to,qo]
            spin_buffer[iw, start + idet, :] += (
                    dot_real_cplx(
                        chol_factor[iw,u,p],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[po,so]*G0[to,uo] - G0[to,so]*G0[po,uo]
            spin_buffer[iw, start + idet, :] -= (
                    dot_real_cplx(
                        chol_factor[iw,q,r],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[po,qo]*G0[to,uo] - G0[to,qo]*G0[po,uo]
            spin_buffer[iw, start + idet, :] += (
                    dot_real_cplx(
                        chol_factor[iw,s,r],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[po,qo]*G0[to,so] - G0[to,qo]*G0[po,so]
            spin_buffer[iw, start + idet, :] -= (
                    dot_real_cplx(
                        chol_factor[iw,u,r],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[po,so]*G0[ro,uo] - G0[ro,so]*G0[po,uo]
            spin_buffer[iw, start + idet, :] += (
                    dot_real_cplx(
                        chol_factor[iw,q,t],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[po,qo]*G0[ro,uo] - G0[ro,qo]*G0[po,uo]
            spin_buffer[iw, start + idet, :] -= (
                    dot_real_cplx(
                        chol_factor[iw,s,t],
                        cofac.real,
                        cofac.imag)
                    )
            cofac = G0[po,qo]*G0[ro,so] - G0[ro,qo]*G0[po,so]
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
        mapping,
        chol_fact,
        buffer,
        det_sls):
    start = det_sls.start
    ndets = cre.shape[0]
    nwalkers = chol_fact.shape[0]
    for iw in range(nwalkers):
        for idet in range(ndets):
            p = mapping[cre[idet, 0]]
            q = anh[idet, 0]
            r = mapping[cre[idet, 1]]
            s = anh[idet, 1]
            buffer[iw, start+idet] += numpy.dot(chol_fact[iw,q,p], chol_fact[iw,s,r]) + 0j
            buffer[iw, start+idet] -= numpy.dot(chol_fact[iw,q,r], chol_fact[iw,s,p]) + 0j

@jit(nopython=True, fastmath=True)
def build_cofactor_matrix(
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
def build_cofactor_matrix_4(
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
        mapping,
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
            p = mapping[ps[idet]]
            spin_buffer[iw, start + idet] += dot_real_cplx(
                                            chol_factor[iw, qs[idet], p],
                                            det_cofactor.real,
                                            det_cofactor.imag,
                                            )

@jit(nopython=True, fastmath=True)
def fill_os_nfold(
        cre,
        anh,
        mapping,
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
            build_cofactor_matrix(
                    iex,
                    jex,
                    det_matrix,
                    cof_mat)
            # nwalkers x ndet
            phase = (-1.0 + 0.j)**(iex + jex)
            reduce_os_spin_factor(
                    ps,
                    qs,
                    mapping,
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
        mapping,
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
            r = mapping[rs[idet]]
            p = mapping[ps[idet]]
            chol_a = chol_factor[iw, ss[idet], r]
            chol_b = chol_factor[iw, qs[idet], p]
            cont_ab = numpy.dot(chol_a, chol_b)
            spin_buffer[iw, start + idet] += dot_real_cplx(
                                            cont_ab,
                                            det_cofactor.real,
                                            det_cofactor.imag,
                                            )
            chol_c = chol_factor[iw, qs[idet], r]
            chol_d = chol_factor[iw, ss[idet], p]
            cont_cd = numpy.dot(chol_c, chol_d)
            spin_buffer[iw, start + idet] -= dot_real_cplx(
                                            cont_cd,
                                            det_cofactor.real,
                                            det_cofactor.imag,
                                            )

def get_ss_nfold(
        cre,
        anh,
        mapping,
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
                    build_cofactor_matrix_4(
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
                            mapping,
                            phase,
                            cof_mat,
                            chol_fact,
                            buffer,
                            det_sls)
