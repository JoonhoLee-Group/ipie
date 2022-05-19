from numba import jit

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
