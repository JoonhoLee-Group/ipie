import cupy


_HUBBARD_SINGLE_SITE_KERNELS = None


def get_hubbard_single_site_kernels():
    global _HUBBARD_SINGLE_SITE_KERNELS
    if _HUBBARD_SINGLE_SITE_KERNELS is not None:
        return _HUBBARD_SINGLE_SITE_KERNELS

    code = r"""
    #include <cuComplex.h>

    __device__ inline cuDoubleComplex c_one() {
        return make_cuDoubleComplex(1.0, 0.0);
    }

    __device__ inline cuDoubleComplex c_conj(cuDoubleComplex z) {
        return make_cuDoubleComplex(cuCreal(z), -cuCimag(z));
    }

    __device__ inline cuDoubleComplex c_scale(cuDoubleComplex z, double a) {
        return make_cuDoubleComplex(a * cuCreal(z), a * cuCimag(z));
    }

    __device__ cuDoubleComplex site_green(
        const cuDoubleComplex* inv,
        const cuDoubleComplex* phi,
        const cuDoubleComplex* psi,
        const int walker,
        const int site,
        const int nocc,
        const int nbasis
    ) {
        cuDoubleComplex g = make_cuDoubleComplex(0.0, 0.0);
        const long inv0 = ((long) walker) * nocc * nocc;
        const long phi0 = ((long) walker) * nbasis * nocc + ((long) site) * nocc;
        for (int j = 0; j < nocc; ++j) {
            cuDoubleComplex q = make_cuDoubleComplex(0.0, 0.0);
            for (int k = 0; k < nocc; ++k) {
                q = cuCadd(q, cuCmul(inv[inv0 + ((long) k) * nocc + j], phi[phi0 + k]));
            }
            g = cuCadd(g, cuCmul(c_conj(psi[j]), q));
        }
        return g;
    }

    extern "C" __global__ void hubbard_site_update_kernel(
        cuDoubleComplex* phia,
        cuDoubleComplex* phib,
        const cuDoubleComplex* inva,
        const cuDoubleComplex* invb,
        const cuDoubleComplex* psi0a,
        const cuDoubleComplex* psi0b,
        const cuDoubleComplex* delta,
        const cuDoubleComplex* aux_wfac,
        const double* random_values,
        double* weight,
        cuDoubleComplex* ovlp,
        cuDoubleComplex* vtup,
        cuDoubleComplex* vtdown,
        int* xi_out,
        signed char* live_out,
        const int site,
        const int nwalkers,
        const int nbasis,
        const int nup,
        const int ndown,
        const int rhf
    ) {
        const int iw = blockIdx.x * blockDim.x + threadIdx.x;
        if (iw >= nwalkers) {
            return;
        }

        const cuDoubleComplex* psi_a_site = psi0a + ((long) site) * nup;
        const cuDoubleComplex* psi_b_site = psi0b + ((long) site) * ndown;

        cuDoubleComplex gup = site_green(inva, phia, psi_a_site, iw, site, nup, nbasis);
        cuDoubleComplex gdown;
        if (ndown > 0 && !rhf) {
            gdown = site_green(invb, phib, psi_b_site, iw, site, ndown, nbasis);
        } else if (ndown > 0 && rhf) {
            gdown = gup;
        } else {
            gdown = make_cuDoubleComplex(0.0, 0.0);
        }

        cuDoubleComplex r1a = cuCadd(c_one(), cuCmul(delta[0], gup));
        cuDoubleComplex r1b = cuCadd(c_one(), cuCmul(delta[1], gdown));
        cuDoubleComplex r2a = cuCadd(c_one(), cuCmul(delta[2], gup));
        cuDoubleComplex r2b = cuCadd(c_one(), cuCmul(delta[3], gdown));
        cuDoubleComplex prob0 = c_scale(cuCmul(cuCmul(r1a, r1b), aux_wfac[0]), 0.5);
        cuDoubleComplex prob1 = c_scale(cuCmul(cuCmul(r2a, r2b), aux_wfac[1]), 0.5);

        const double prob0_real = cuCreal(prob0);
        const double prob1_real = cuCreal(prob1);
        const double p0_real = prob0_real > 0.0 ? prob0_real : 0.0;
        const double p1_real = prob1_real > 0.0 ? prob1_real : 0.0;
        const double norm = p0_real + p1_real;
        const double abs_weight = weight[iw] >= 0.0 ? weight[iw] : -weight[iw];
        const bool live = (norm > 0.0) && (abs_weight > 0.0);
        const double p0 = live ? p0_real / norm : 1.0;
        const int xi = (random_values[iw] >= p0) ? 1 : 0;
        const cuDoubleComplex selected = (xi == 0) ? prob0 : prob1;

        xi_out[iw] = xi;
        live_out[iw] = live ? 1 : 0;
        weight[iw] = live ? weight[iw] * norm : 0.0;
        if (live) {
            ovlp[iw] = cuCmul(c_scale(ovlp[iw], 2.0), selected);
        }

        const cuDoubleComplex delta_up = delta[2 * xi];
        const cuDoubleComplex delta_dn = delta[2 * xi + 1];
        const long phia0 = ((long) iw) * nbasis * nup + ((long) site) * nup;
        const long vtup0 = ((long) iw) * nup;
        for (int j = 0; j < nup; ++j) {
            cuDoubleComplex vt = live ? cuCmul(phia[phia0 + j], delta_up)
                                      : make_cuDoubleComplex(0.0, 0.0);
            vtup[vtup0 + j] = vt;
            phia[phia0 + j] = cuCadd(phia[phia0 + j], vt);
        }

        if (ndown > 0 && !rhf) {
            const long phib0 = ((long) iw) * nbasis * ndown + ((long) site) * ndown;
            const long vtdown0 = ((long) iw) * ndown;
            for (int j = 0; j < ndown; ++j) {
                cuDoubleComplex vt = live ? cuCmul(phib[phib0 + j], delta_dn)
                                          : make_cuDoubleComplex(0.0, 0.0);
                vtdown[vtdown0 + j] = vt;
                phib[phib0 + j] = cuCadd(phib[phib0 + j], vt);
            }
        }
    }

    extern "C" __global__ void hubbard_site_update_parallel_kernel(
        cuDoubleComplex* phia,
        cuDoubleComplex* phib,
        const cuDoubleComplex* inva,
        const cuDoubleComplex* invb,
        const cuDoubleComplex* psi0a,
        const cuDoubleComplex* psi0b,
        const cuDoubleComplex* delta,
        const cuDoubleComplex* aux_wfac,
        const double* random_values,
        double* weight,
        cuDoubleComplex* ovlp,
        cuDoubleComplex* vtup,
        cuDoubleComplex* vtdown,
        int* xi_out,
        signed char* live_out,
        const int site,
        const int nwalkers,
        const int nbasis,
        const int nup,
        const int ndown,
        const int rhf
    ) {
        const int iw = blockIdx.x;
        const int tid = threadIdx.x;
        if (iw >= nwalkers) {
            return;
        }

        extern __shared__ double raw_shared_d[];
        double* gup_re = raw_shared_d;
        double* gup_im = gup_re + blockDim.x;
        double* gdn_re = gup_im + blockDim.x;
        double* gdn_im = gdn_re + blockDim.x;

        const cuDoubleComplex* psi_a_site = psi0a + ((long) site) * nup;
        const cuDoubleComplex* psi_b_site = psi0b + ((long) site) * ndown;
        const long inva0 = ((long) iw) * nup * nup;
        const long phia0 = ((long) iw) * nbasis * nup + ((long) site) * nup;
        cuDoubleComplex gup = make_cuDoubleComplex(0.0, 0.0);
        for (int idx = tid; idx < nup * nup; idx += blockDim.x) {
            const int j = idx / nup;
            const int k = idx - j * nup;
            cuDoubleComplex term =
                cuCmul(cuCmul(c_conj(psi_a_site[j]), inva[inva0 + ((long) k) * nup + j]),
                       phia[phia0 + k]);
            gup = cuCadd(gup, term);
        }

        cuDoubleComplex gdown = make_cuDoubleComplex(0.0, 0.0);
        if (ndown > 0 && !rhf) {
            const long invb0 = ((long) iw) * ndown * ndown;
            const long phib0 = ((long) iw) * nbasis * ndown + ((long) site) * ndown;
            for (int idx = tid; idx < ndown * ndown; idx += blockDim.x) {
                const int j = idx / ndown;
                const int k = idx - j * ndown;
                cuDoubleComplex term =
                    cuCmul(cuCmul(c_conj(psi_b_site[j]), invb[invb0 + ((long) k) * ndown + j]),
                           phib[phib0 + k]);
                gdown = cuCadd(gdown, term);
            }
        }

        gup_re[tid] = cuCreal(gup);
        gup_im[tid] = cuCimag(gup);
        gdn_re[tid] = cuCreal(gdown);
        gdn_im[tid] = cuCimag(gdown);
        __syncthreads();

        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride) {
                gup_re[tid] += gup_re[tid + stride];
                gup_im[tid] += gup_im[tid + stride];
                gdn_re[tid] += gdn_re[tid + stride];
                gdn_im[tid] += gdn_im[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            gup = make_cuDoubleComplex(gup_re[0], gup_im[0]);
            if (ndown > 0 && rhf) {
                gdown = gup;
            } else {
                gdown = make_cuDoubleComplex(gdn_re[0], gdn_im[0]);
            }

            cuDoubleComplex r1a = cuCadd(c_one(), cuCmul(delta[0], gup));
            cuDoubleComplex r1b = cuCadd(c_one(), cuCmul(delta[1], gdown));
            cuDoubleComplex r2a = cuCadd(c_one(), cuCmul(delta[2], gup));
            cuDoubleComplex r2b = cuCadd(c_one(), cuCmul(delta[3], gdown));
            cuDoubleComplex prob0 = c_scale(cuCmul(cuCmul(r1a, r1b), aux_wfac[0]), 0.5);
            cuDoubleComplex prob1 = c_scale(cuCmul(cuCmul(r2a, r2b), aux_wfac[1]), 0.5);

            const double prob0_real = cuCreal(prob0);
            const double prob1_real = cuCreal(prob1);
            const double p0_real = prob0_real > 0.0 ? prob0_real : 0.0;
            const double p1_real = prob1_real > 0.0 ? prob1_real : 0.0;
            const double norm = p0_real + p1_real;
            const double abs_weight = weight[iw] >= 0.0 ? weight[iw] : -weight[iw];
            const bool live = (norm > 0.0) && (abs_weight > 0.0);
            const double p0 = live ? p0_real / norm : 1.0;
            const int xi = (random_values[iw] >= p0) ? 1 : 0;
            const cuDoubleComplex selected = (xi == 0) ? prob0 : prob1;

            xi_out[iw] = xi;
            live_out[iw] = live ? 1 : 0;
            weight[iw] = live ? weight[iw] * norm : 0.0;
            if (live) {
                ovlp[iw] = cuCmul(c_scale(ovlp[iw], 2.0), selected);
            }
        }
        __syncthreads();

        const int xi = xi_out[iw];
        const bool live = live_out[iw] != 0;
        const cuDoubleComplex delta_up = delta[2 * xi];
        const cuDoubleComplex delta_dn = delta[2 * xi + 1];
        const long vtup0 = ((long) iw) * nup;
        for (int j = tid; j < nup; j += blockDim.x) {
            cuDoubleComplex vt = live ? cuCmul(phia[phia0 + j], delta_up)
                                      : make_cuDoubleComplex(0.0, 0.0);
            vtup[vtup0 + j] = vt;
            phia[phia0 + j] = cuCadd(phia[phia0 + j], vt);
        }

        if (ndown > 0 && !rhf) {
            const long phib0 = ((long) iw) * nbasis * ndown + ((long) site) * ndown;
            const long vtdown0 = ((long) iw) * ndown;
            for (int j = tid; j < ndown; j += blockDim.x) {
                cuDoubleComplex vt = live ? cuCmul(phib[phib0 + j], delta_dn)
                                          : make_cuDoubleComplex(0.0, 0.0);
                vtdown[vtdown0 + j] = vt;
                phib[phib0 + j] = cuCadd(phib[phib0 + j], vt);
            }
        }
    }

    extern "C" __global__ void sherman_morrison_update_kernel(
        cuDoubleComplex* inv,
        const cuDoubleComplex* psi_site,
        const cuDoubleComplex* vt,
        const int nwalkers,
        const int nocc
    ) {
        const int iw = blockIdx.x;
        const int tid = threadIdx.x;
        if (iw >= nwalkers) {
            return;
        }

        extern __shared__ unsigned char raw_shared[];
        cuDoubleComplex* au = reinterpret_cast<cuDoubleComplex*>(raw_shared);
        cuDoubleComplex* vta = au + nocc;
        cuDoubleComplex* denom_shared = vta + nocc;

        const long inv0 = ((long) iw) * nocc * nocc;
        const long vt0 = ((long) iw) * nocc;
        for (int row = tid; row < nocc; row += blockDim.x) {
            cuDoubleComplex au_row = make_cuDoubleComplex(0.0, 0.0);
            cuDoubleComplex vta_row = make_cuDoubleComplex(0.0, 0.0);
            for (int k = 0; k < nocc; ++k) {
                const cuDoubleComplex u_k = c_conj(psi_site[k]);
                au_row = cuCadd(au_row, cuCmul(inv[inv0 + ((long) row) * nocc + k], u_k));
                vta_row = cuCadd(vta_row, cuCmul(vt[vt0 + k], inv[inv0 + ((long) k) * nocc + row]));
            }
            au[row] = au_row;
            vta[row] = vta_row;
        }
        __syncthreads();

        if (tid == 0) {
            cuDoubleComplex denom = c_one();
            for (int k = 0; k < nocc; ++k) {
                denom = cuCadd(denom, cuCmul(vta[k], c_conj(psi_site[k])));
            }
            denom_shared[0] = denom;
        }
        __syncthreads();

        const cuDoubleComplex denom = denom_shared[0];
        for (int row = tid; row < nocc; row += blockDim.x) {
            for (int col = 0; col < nocc; ++col) {
                const cuDoubleComplex update = cuCdiv(cuCmul(au[row], vta[col]), denom);
                inv[inv0 + ((long) row) * nocc + col] =
                    cuCsub(inv[inv0 + ((long) row) * nocc + col], update);
            }
        }
    }

    extern "C" __global__ void sherman_morrison_apply_large_kernel(
        cuDoubleComplex* inv,
        const cuDoubleComplex* au,
        const cuDoubleComplex* vta,
        const cuDoubleComplex* denom,
        const int nwalkers,
        const int nocc
    ) {
        const int iw = blockIdx.z;
        const int col = blockIdx.x * blockDim.x + threadIdx.x;
        const int row = blockIdx.y * blockDim.y + threadIdx.y;
        if (iw >= nwalkers) {
            return;
        }

        extern __shared__ unsigned char raw_shared_apply[];
        cuDoubleComplex* au_tile = reinterpret_cast<cuDoubleComplex*>(raw_shared_apply);
        cuDoubleComplex* vta_tile = au_tile + blockDim.y;
        cuDoubleComplex* inv_denom = vta_tile + blockDim.x;

        const long aux0 = ((long) iw) * nocc;
        if (threadIdx.x == 0 && row < nocc) {
            au_tile[threadIdx.y] = au[aux0 + row];
        }
        if (threadIdx.y == 0 && col < nocc) {
            vta_tile[threadIdx.x] = vta[aux0 + col];
        }
        if (threadIdx.x == 0 && threadIdx.y == 0) {
            inv_denom[0] = cuCdiv(c_one(), denom[iw]);
        }
        __syncthreads();

        if (row >= nocc || col >= nocc) {
            return;
        }

        const long inv0 = ((long) iw) * nocc * nocc;
        const cuDoubleComplex update =
            cuCmul(cuCmul(au_tile[threadIdx.y], vta_tile[threadIdx.x]), inv_denom[0]);
        inv[inv0 + ((long) row) * nocc + col] =
            cuCsub(inv[inv0 + ((long) row) * nocc + col], update);
    }
    """
    _HUBBARD_SINGLE_SITE_KERNELS = {
        "site_update": cupy.RawKernel(code, "hubbard_site_update_kernel"),
        "site_update_parallel": cupy.RawKernel(code, "hubbard_site_update_parallel_kernel"),
        "sherman_morrison": cupy.RawKernel(code, "sherman_morrison_update_kernel"),
        "sherman_morrison_apply_large": cupy.RawKernel(
            code, "sherman_morrison_apply_large_kernel"
        ),
    }
    return _HUBBARD_SINGLE_SITE_KERNELS
