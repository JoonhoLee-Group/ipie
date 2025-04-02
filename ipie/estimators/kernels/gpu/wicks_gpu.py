# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Fionn Malone <fionn.malone@gmail.com>
#

import numpy
import cupy  # pylint: disable=import-error

# Overlap

# Note mapping arrays account for occupied indices not matching compressed
# format which may arise when the reference determinant does not follow aufbau
# principle (i.e. not doubly occupied up to the fermi level.
# e.g. D0a = [0, 1, 4], mapping = [0, 1, 0, 0, 2]
# mapping[orb] is then used to address arrays of dimension nocc * nmo and
# similar (half rotated Green's functio) and avoid out of bounds errors.


def get_dets_singles(cre, anh, mapping, offset, G0, dets):
    """Get overlap from singly excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    dets : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    qs = cupy.ascontiguousarray(anh[:, 0]) + offset
    ndets = qs.shape[0]
    nwalkers = G0.shape[0]

    get_dets_singles_kernel = cupy.RawKernel(
        r"""
            #include<cuComplex.h>
            extern "C" __global__
            void get_dets_singles_kernel(int* cre, int* qs, int* mapping, int offset, int nex, int nwalkers, int ndets, int G0_dim2, int G0_dim3, cuDoubleComplex* G0, cuDoubleComplex* dets){  
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = gridDim.x * blockDim.x;
                int iw, idet;
                int p;
                size_t max_size;
                max_size = nwalkers*ndets;

                for (int thread = idx; thread < max_size; thread += stride){
                    iw = thread/ndets;
                    
                    idet = thread%ndets;
                
                    p = mapping[cre[idet*nex]] + offset;
                    dets[iw*ndets + idet] = G0[iw*G0_dim2*G0_dim3 + p*G0_dim3 + qs[idet]];
 
                }
                
            }   
            """,
        "get_dets_singles_kernel",
    )

    get_dets_singles_kernel(
        (int(numpy.ceil(ndets * nwalkers / 64)),),
        (64,),
        (
            cre,
            qs,
            mapping,
            offset,
            cre.shape[1],
            nwalkers,
            ndets,
            G0.shape[1],
            G0.shape[2],
            G0,
            dets,
        ),
    )


def get_dets_doubles(cre, anh, mapping, offset, G0, dets):
    """Get overlap from double excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    dets : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    qs = cupy.ascontiguousarray(anh[:, 0]) + offset
    ss = cupy.ascontiguousarray(anh[:, 1]) + offset
    ndets = qs.shape[0]
    nwalkers = G0.shape[0]

    get_dets_doubles_kernel = cupy.RawKernel(
        r"""
            #include<cuComplex.h>
            extern "C" __global__
            void get_dets_doubles_kernel(int* cre, int* qs, int* ss, int* mapping, int offset, int nex, int nwalkers, int ndets, int G0_dim2, int G0_dim3, cuDoubleComplex* G0, cuDoubleComplex* dets){  
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = gridDim.x * blockDim.x;
                int iw, idet;
                int p, r;
                size_t max_size;
                max_size = nwalkers*ndets;

                for (int thread = idx; thread < max_size; thread += stride){
                    iw = thread/ndets;
                    
                    idet = thread%ndets;
                
                    p = mapping[cre[idet*nex]] + offset;
                    r = mapping[cre[idet*nex+1]] + offset;
                    dets[iw*ndets + idet] = cuCsub(cuCmul(G0[iw*G0_dim2*G0_dim3 + p*G0_dim3 + qs[idet]], G0[iw*G0_dim2*G0_dim3 + r*G0_dim3 + ss[idet]]), cuCmul(G0[iw*G0_dim2*G0_dim3 + p*G0_dim3 + ss[idet]], G0[iw*G0_dim2*G0_dim3 + r*G0_dim3 + qs[idet]]));
 
                }
                
            }   
            """,
        "get_dets_doubles_kernel",
    )

    get_dets_doubles_kernel(
        (int(numpy.ceil(ndets * nwalkers / 64)),),
        (64,),
        (
            cre,
            qs,
            ss,
            mapping,
            offset,
            cre.shape[1],
            nwalkers,
            ndets,
            G0.shape[1],
            G0.shape[2],
            G0,
            dets,
        ),
    )


def get_dets_triples(cre, anh, mapping, offset, G0, dets):
    """Get overlap from double excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    dets : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    ndets = len(cre)
    nwalkers = G0.shape[0]

    get_dets_triples_kernel = cupy.RawKernel(
        r"""
            #include<cuComplex.h>
            extern "C" __global__
            void get_dets_triples_kernel(int* cre, int* anh, int* mapping, int offset, int nex, int nwalkers, int ndets, int G0_dim2, int G0_dim3, cuDoubleComplex* G0, cuDoubleComplex* dets){  
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = gridDim.x * blockDim.x;
                int iw, idet;
                int ps, qs, rs, ss, us, ts;
                size_t max_size;
                max_size = nwalkers*ndets;

                for (int thread = idx; thread < max_size; thread += stride){
                    iw = thread/ndets;
                    
                    idet = thread%ndets;
                
                    ps = mapping[cre[idet*nex]] + offset;
                    qs = anh[idet*nex] + offset;
                    rs = mapping[cre[idet*nex+1]] + offset;
                    ss = anh[idet*nex+1] + offset;
                    ts = mapping[cre[idet*nex+2]] + offset;
                    us = anh[idet*nex+2] + offset;
                    dets[iw*ndets + idet] = cuCadd(
                    cuCsub(
                    cuCmul(G0[iw*G0_dim2*G0_dim3 + ps*G0_dim3 + qs],cuCsub(cuCmul(G0[iw*G0_dim2*G0_dim3 + rs*G0_dim3 + ss], G0[iw*G0_dim2*G0_dim3 + ts*G0_dim3 + us]), cuCmul(G0[iw*G0_dim2*G0_dim3 + rs*G0_dim3 + us], G0[iw*G0_dim2*G0_dim3 + ts*G0_dim3 + ss])))
                    ,cuCmul(G0[iw*G0_dim2*G0_dim3 + ps*G0_dim3 + ss],cuCsub(cuCmul(G0[iw*G0_dim2*G0_dim3 + rs*G0_dim3 + qs], G0[iw*G0_dim2*G0_dim3 + ts*G0_dim3 + us]), cuCmul(G0[iw*G0_dim2*G0_dim3 + rs*G0_dim3 + us], G0[iw*G0_dim2*G0_dim3 + ts*G0_dim3 + qs])))
                    ),
                    cuCmul(G0[iw*G0_dim2*G0_dim3 + ps*G0_dim3 + us],cuCsub(cuCmul(G0[iw*G0_dim2*G0_dim3 + rs*G0_dim3 + qs], G0[iw*G0_dim2*G0_dim3 + ts*G0_dim3 + ss]), cuCmul(G0[iw*G0_dim2*G0_dim3 + rs*G0_dim3 + ss], G0[iw*G0_dim2*G0_dim3 + ts*G0_dim3 + qs])))
                    );
                }
                
            }   
            """,
        "get_dets_triples_kernel",
    )

    get_dets_triples_kernel(
        (int(numpy.ceil(ndets * nwalkers / 64)),),
        (64,),
        (
            cre,
            anh,
            mapping,
            offset,
            cre.shape[1],
            nwalkers,
            ndets,
            G0.shape[1],
            G0.shape[2],
            G0,
            dets,
        ),
    )


def get_dets_nfold(cre, anh, mapping, offset, G0):
    """Get overlap from n-fold excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    dets : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """

    ndets = len(cre)
    nwalkers = G0.shape[0]
    nex = cre.shape[-1]
    det = cupy.zeros((nwalkers, ndets, nex, nex), dtype=numpy.complex128)

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    get_dets_nfold_kernel = cupy.RawKernel(
        r"""
            #include<cuComplex.h>
            extern "C" __global__
            void get_dets_nfold_kernel(int* cre, int* anh, int* mapping, int offset, int nex, int nwalkers, int ndets, int G0_dim2, int G0_dim3, cuDoubleComplex* G0, cuDoubleComplex* det){  
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = gridDim.x * blockDim.x;
                int iw, idet;
                int p, q, r, s, iex, jex;
                size_t max_size;
                max_size = ndets*nwalkers;

                for (int thread = idx; thread < max_size; thread += stride){
                    iw = thread/ndets;

                    idet =  thread%ndets;
                    
                    for (iex = 0; iex < nex; iex++){
                        p = mapping[cre[idet*nex + iex]] + offset;
                        q = anh[idet*nex + iex] + offset;
                        det[iw*ndets*nex*nex + idet*nex*nex + iex*nex + iex] = G0[iw*G0_dim2*G0_dim3 + p*G0_dim3 + q];
                        for (jex = iex + 1; jex < nex; jex++){
                            r = mapping[cre[idet*nex + jex]] + offset;
                            s = anh[idet*nex + jex] + offset;
                            det[iw*ndets*nex*nex + idet*nex*nex + iex*nex + jex] = G0[iw*G0_dim2*G0_dim3 + p*G0_dim3 + s];
                            det[iw*ndets*nex*nex + idet*nex*nex + jex*nex + iex] = G0[iw*G0_dim2*G0_dim3 + r*G0_dim3 + q];
                        }
                    }
                }
                
            }   
            """,
        "get_dets_nfold_kernel",
    )

    get_dets_nfold_kernel(
        (int(numpy.ceil(ndets * nwalkers / 64)),),
        (64,),
        (cre, anh, mapping, offset, nex, nwalkers, ndets, G0.shape[1], G0.shape[2], G0, det),
    )

    dets = cupy.linalg.det(det)

    return dets


def build_det_matrix(cre, anh, mapping, offset, G0, det_mat):
    """Build matrix of determinants for n-fold excitations.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    det_matrix : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """

    nwalkers = det_mat.shape[0]
    ndets = det_mat.shape[1]
    if ndets == 0:
        return

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    nex = det_mat.shape[2]

    build_det_matrix_kernel = cupy.RawKernel(
        r"""
            #include<cuComplex.h>
            extern "C" __global__
            void build_det_matrix_kernel(int* cre, int* anh, int* mapping, int offset, int nex, int nwalkers, int ndets, int G0_dim2, int G0_dim3, cuDoubleComplex* G0, cuDoubleComplex* det){  
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = gridDim.x * blockDim.x;
                int iw, idet;
                int p, q, r, s, iex, jex;
                size_t max_size;
                max_size = ndets*nwalkers;

                for (int thread = idx; thread < max_size; thread += stride){
                    iw = thread/ndets;

                    idet =  thread%ndets;
                    
                    for (iex = 0; iex < nex; iex++){
                        p = mapping[cre[idet*nex + iex]] + offset;
                        q = anh[idet*nex + iex] + offset;
                        det[iw*ndets*nex*nex + idet*nex*nex + iex*nex + iex] = G0[iw*G0_dim2*G0_dim3 + p*G0_dim3 + q];
                        for (jex = iex + 1; jex < nex; jex++){
                            r = mapping[cre[idet*nex + jex]] + offset;
                            s = anh[idet*nex + jex] + offset;
                            det[iw*ndets*nex*nex + idet*nex*nex + iex*nex + jex] = G0[iw*G0_dim2*G0_dim3 + p*G0_dim3 + s];
                            det[iw*ndets*nex*nex + idet*nex*nex + jex*nex + iex] = G0[iw*G0_dim2*G0_dim3 + r*G0_dim3 + q];
                        }
                    }
                }
                
            }   
            """,
        "build_det_matrix_kernel",
    )

    build_det_matrix_kernel(
        (int(numpy.ceil(ndets * nwalkers / 64)),),
        (64,),
        (cre, anh, mapping, offset, nex, nwalkers, ndets, G0.shape[1], G0.shape[2], G0, det_mat),
    )


def reduce_CI_singles(cre, anh, mapping, phases, CI):
    """Reduction to CI intermediate for singles.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    phases : np.ndarray
        Phase factors.
    CI : np.ndarray
        Output array for CI intermediate.

    Returns
    -------
    None
    """

    ndets = len(cre)
    nwalkers = phases.shape[0]

    phases = cupy.ascontiguousarray(phases)

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    ps = cupy.ascontiguousarray(cre[:, 0])
    qs = cupy.ascontiguousarray(anh[:, 0])

    reduce_CI_singles_kernel = cupy.RawKernel(
        r"""
            #include<cuComplex.h>
            extern "C" __global__
            void reduce_CI_singles_kernel(int* ps, int* qs, int* mapping, int nwalkers, int ndets, int CI_dim2, int CI_dim3, cuDoubleComplex* phases, cuDoubleComplex* CI){  
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = gridDim.x * blockDim.x;
                int iw, idet;
                int p, q;
                size_t max_size;
                max_size = ndets*nwalkers;

                for (int thread = idx; thread < max_size; thread += stride){
                    iw = thread/ndets;

                    idet =  thread%ndets;
                    
                    p = mapping[ps[idet]];
                    q = qs[idet];
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + p), phases[iw*ndets + idet].x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + p)+1, phases[iw*ndets + idet].y);
                    
                }
                
            }   
            """,
        "reduce_CI_singles_kernel",
    )

    reduce_CI_singles_kernel(
        (int(numpy.ceil(ndets * nwalkers / 64)),),
        (64,),
        (ps, qs, mapping, nwalkers, ndets, CI.shape[1], CI.shape[2], phases, CI),
    )


def reduce_CI_doubles(cre, anh, mapping, offset, phases, G0, CI):
    """Reduction to CI intermediate for triples.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    phases : np.ndarray
        Phase factors.
    CI : np.ndarray
        Output array for CI intermediate.

    Returns
    -------
    None
    """

    ndets = len(cre)
    nwalkers = G0.shape[0]

    phases = cupy.ascontiguousarray(phases)

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    ps = cupy.ascontiguousarray(cre[:, 0])
    qs = cupy.ascontiguousarray(anh[:, 0])
    rs = cupy.ascontiguousarray(cre[:, 1])
    ss = cupy.ascontiguousarray(anh[:, 1])

    reduce_CI_doubles_kernel = cupy.RawKernel(
        r"""
            #include<cuComplex.h>
            extern "C" __global__
            void reduce_CI_doubles_kernel(int* ps, int* qs, int* rs, int* ss, int* mapping, int offset, int nwalkers, int ndets, int CI_dim2, int CI_dim3, int G0_dim2, int G0_dim3, cuDoubleComplex* phases, cuDoubleComplex* G0, cuDoubleComplex* CI){  
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = gridDim.x * blockDim.x;
                int iw, idet;
                int p, q, r, s, po, qo, ro, so;
                size_t max_size;
                max_size = ndets*nwalkers;

                for (int thread = idx; thread < max_size; thread += stride){
                    iw = thread/ndets;

                    idet =  thread%ndets;
                    
                    p = mapping[ps[idet]];
                    q = qs[idet];
                    r = mapping[rs[idet]];
                    s = ss[idet];
                    po = mapping[ps[idet]] + offset;
                    qo = qs[idet] + offset;
                    ro = mapping[rs[idet]] + offset;
                    so = ss[idet] + offset;
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + p), cuCmul(G0[iw*G0_dim2*G0_dim3 + ro*G0_dim3 + so],phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + p)+1, cuCmul(G0[iw*G0_dim2*G0_dim3 + ro*G0_dim3 + so],phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + s*CI_dim3 + r), cuCmul(G0[iw*G0_dim2*G0_dim3 + po*G0_dim3 + qo],phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + s*CI_dim3 + r)+1, cuCmul(G0[iw*G0_dim2*G0_dim3 + po*G0_dim3 + qo],phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + r), -cuCmul(G0[iw*G0_dim2*G0_dim3 + po*G0_dim3 + so],phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + r)+1, -cuCmul(G0[iw*G0_dim2*G0_dim3 + po*G0_dim3 + so],phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + s*CI_dim3 + p), -cuCmul(G0[iw*G0_dim2*G0_dim3 + ro*G0_dim3 + qo],phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + s*CI_dim3 + p)+1, -cuCmul(G0[iw*G0_dim2*G0_dim3 + ro*G0_dim3 + qo],phases[iw*ndets + idet]).y);
                    
                }
                
            }   
            """,
        "reduce_CI_doubles_kernel",
    )

    reduce_CI_doubles_kernel(
        (int(numpy.ceil(ndets * nwalkers / 64)),),
        (64,),
        (
            ps,
            qs,
            rs,
            ss,
            mapping,
            offset,
            nwalkers,
            ndets,
            CI.shape[1],
            CI.shape[2],
            G0.shape[1],
            G0.shape[2],
            phases,
            G0,
            CI,
        ),
    )


def reduce_CI_triples(cre, anh, mapping, offset, phases, G0, CI):
    """Reduction to CI intermediate for triples.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    phases : np.ndarray
        Phase factors.
    CI : np.ndarray
        Output array for CI intermediate.

    Returns
    -------
    None
    """

    ndets = len(cre)
    nwalkers = G0.shape[0]

    phases = cupy.ascontiguousarray(phases)

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    ps = cupy.ascontiguousarray(cre[:, 0])
    qs = cupy.ascontiguousarray(anh[:, 0])
    rs = cupy.ascontiguousarray(cre[:, 1])
    ss = cupy.ascontiguousarray(anh[:, 1])
    ts = cupy.ascontiguousarray(cre[:, 2])
    us = cupy.ascontiguousarray(anh[:, 2])

    reduce_CI_triples_kernel = cupy.RawKernel(
        r"""
            #include<cuComplex.h>
            extern "C" __global__
            void reduce_CI_triples_kernel(int* ps, int* qs, int* rs, int* ss, int* ts, int* us, int* mapping, int offset, int nwalkers, int ndets, int CI_dim2, int CI_dim3, int G0_dim2, int G0_dim3, cuDoubleComplex* phases, cuDoubleComplex* G0, cuDoubleComplex* CI){  
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = gridDim.x * blockDim.x;
                int iw, idet;
                int p, q, r, s, t, u, po, qo, ro, so, to, uo, G0_idx_rs, G0_idx_tu, G0_idx_ru, G0_idx_ts, G0_idx_rq, G0_idx_tq, G0_idx_ps, G0_idx_pu, G0_idx_pq;
                size_t max_size;
                max_size = ndets*nwalkers;

                for (int thread = idx; thread < max_size; thread += stride){
                    iw = thread/ndets;

                    idet =  thread%ndets;
                    
                    p = mapping[ps[idet]];
                    q = qs[idet];
                    r = mapping[rs[idet]];
                    s = ss[idet];
                    t = mapping[ts[idet]];
                    u = us[idet];
                    po = mapping[ps[idet]] + offset;
                    qo = qs[idet] + offset;
                    ro = mapping[rs[idet]] + offset;
                    so = ss[idet] + offset;
                    to = mapping[ts[idet]] + offset;
                    uo = us[idet] + offset;
                    
                    G0_idx_rs = iw*G0_dim2*G0_dim3 + ro*G0_dim3 + so;
                    G0_idx_tu = iw*G0_dim2*G0_dim3 + to*G0_dim3 + uo;
                    G0_idx_ru = iw*G0_dim2*G0_dim3 + ro*G0_dim3 + uo;
                    G0_idx_ts = iw*G0_dim2*G0_dim3 + to*G0_dim3 + so;
                    G0_idx_rq = iw*G0_dim2*G0_dim3 + ro*G0_dim3 + qo;
                    G0_idx_tq = iw*G0_dim2*G0_dim3 + to*G0_dim3 + qo;
                    G0_idx_ps = iw*G0_dim2*G0_dim3 + po*G0_dim3 + so;
                    G0_idx_pu = iw*G0_dim2*G0_dim3 + po*G0_dim3 + uo;
                    G0_idx_pq = iw*G0_dim2*G0_dim3 + po*G0_dim3 + qo;
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + p), cuCmul(cuCsub(cuCmul(G0[G0_idx_rs],G0[G0_idx_tu]),cuCmul(G0[G0_idx_ru],G0[G0_idx_ts])),phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + p)+1, cuCmul(cuCsub(cuCmul(G0[G0_idx_rs],G0[G0_idx_tu]),cuCmul(G0[G0_idx_ru],G0[G0_idx_ts])),phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + s*CI_dim3 + p), -cuCmul(cuCsub(cuCmul(G0[G0_idx_rq],G0[G0_idx_tu]),cuCmul(G0[G0_idx_ru],G0[G0_idx_tq])),phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + s*CI_dim3 + p)+1, -cuCmul(cuCsub(cuCmul(G0[G0_idx_rq],G0[G0_idx_tu]),cuCmul(G0[G0_idx_ru],G0[G0_idx_tq])),phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + u*CI_dim3 + p), cuCmul(cuCsub(cuCmul(G0[G0_idx_rq],G0[G0_idx_ts]),cuCmul(G0[G0_idx_rs],G0[G0_idx_tq])),phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + u*CI_dim3 + p)+1, cuCmul(cuCsub(cuCmul(G0[G0_idx_rq],G0[G0_idx_ts]),cuCmul(G0[G0_idx_rs],G0[G0_idx_tq])),phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + r), -cuCmul(cuCsub(cuCmul(G0[G0_idx_ps],G0[G0_idx_tu]),cuCmul(G0[G0_idx_pu],G0[G0_idx_ts])),phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + r)+1, -cuCmul(cuCsub(cuCmul(G0[G0_idx_ps],G0[G0_idx_tu]),cuCmul(G0[G0_idx_pu],G0[G0_idx_ts])),phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + s*CI_dim3 + r), cuCmul(cuCsub(cuCmul(G0[G0_idx_pq],G0[G0_idx_tu]),cuCmul(G0[G0_idx_pu],G0[G0_idx_tq])),phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + s*CI_dim3 + r)+1, cuCmul(cuCsub(cuCmul(G0[G0_idx_pq],G0[G0_idx_tu]),cuCmul(G0[G0_idx_pu],G0[G0_idx_tq])),phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + u*CI_dim3 + r), -cuCmul(cuCsub(cuCmul(G0[G0_idx_pq],G0[G0_idx_ts]),cuCmul(G0[G0_idx_ps],G0[G0_idx_tq])),phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + u*CI_dim3 + r)+1, -cuCmul(cuCsub(cuCmul(G0[G0_idx_pq],G0[G0_idx_ts]),cuCmul(G0[G0_idx_ps],G0[G0_idx_tq])),phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + t), cuCmul(cuCsub(cuCmul(G0[G0_idx_ps],G0[G0_idx_ru]),cuCmul(G0[G0_idx_pu],G0[G0_idx_rs])),phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + t)+1, cuCmul(cuCsub(cuCmul(G0[G0_idx_ps],G0[G0_idx_ru]),cuCmul(G0[G0_idx_pu],G0[G0_idx_rs])),phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + s*CI_dim3 + t), -cuCmul(cuCsub(cuCmul(G0[G0_idx_pq],G0[G0_idx_ru]),cuCmul(G0[G0_idx_pu],G0[G0_idx_rq])),phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + s*CI_dim3 + t)+1, -cuCmul(cuCsub(cuCmul(G0[G0_idx_pq],G0[G0_idx_ru]),cuCmul(G0[G0_idx_pu],G0[G0_idx_rq])),phases[iw*ndets + idet]).y);
                    
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + u*CI_dim3 + t), cuCmul(cuCsub(cuCmul(G0[G0_idx_pq],G0[G0_idx_rs]),cuCmul(G0[G0_idx_ps],G0[G0_idx_rq])),phases[iw*ndets + idet]).x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + u*CI_dim3 + t)+1, cuCmul(cuCsub(cuCmul(G0[G0_idx_pq],G0[G0_idx_rs]),cuCmul(G0[G0_idx_ps],G0[G0_idx_rq])),phases[iw*ndets + idet]).y);
                
                }
                
            }   
            """,
        "reduce_CI_triples_kernel",
    )

    reduce_CI_triples_kernel(
        (int(numpy.ceil(ndets * nwalkers / 64)),),
        (64,),
        (
            ps,
            qs,
            rs,
            ss,
            ts,
            us,
            mapping,
            offset,
            nwalkers,
            ndets,
            CI.shape[1],
            CI.shape[2],
            G0.shape[1],
            G0.shape[2],
            phases,
            G0,
            CI,
        ),
    )


def reduce_CI_nfold(cre, anh, mapping, offset, phases, det_mat, cof_mat, CI):
    """Reduction to CI intermediate for n-fold excitations.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    phases : np.ndarray
        Phase factors.
    det_mat: np.ndarray
        Array of determinants <D_I|phi>.
    cof_mat: np.ndarray
        Cofactor matrix previously constructed.
    CI : np.ndarray
        Output array for CI intermediate.

    Returns
    -------
    None
    """
    nexcit = det_mat.shape[-1]
    # phases = cupy.asarray(phases)
    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    phases = cupy.ascontiguousarray(phases)

    nwalkers = cof_mat.shape[0]
    ndets = cof_mat.shape[1]

    cof_mat_all = cupy.zeros(
        (nexcit, nexcit, cof_mat.shape[0], cof_mat.shape[1], cof_mat.shape[2], cof_mat.shape[3]),
        dtype=numpy.complex128,
    )

    for iex in range(nexcit):
        for jex in range(nexcit):
            build_cofactor_matrix_gpu(iex, jex, det_mat, cof_mat)
            cof_mat_all[iex, jex, :, :, :, :] = cof_mat.copy()

    det = cupy.linalg.det(cof_mat_all)

    rhs = cupy.zeros_like(det)

    reduce_nfold_cofac_kernel2 = cupy.RawKernel(
        r"""
            #include<cuComplex.h>
            #include<cuda_runtime.h>
            extern "C" __global__
            void reduce_nfold_cofac_kernel2(int nexcit, int* cre, int* anh, int* mapping, int nwalkers, int ndets, int CI_dim2, int CI_dim3, cuDoubleComplex* phases, cuDoubleComplex* det, cuDoubleComplex* mul_tmp, cuDoubleComplex* CI){  
                int idx = threadIdx.x + blockIdx.x * blockDim.x;
                int stride = gridDim.x * blockDim.x;
                int iex, jex, iw, idet;
                cuDoubleComplex sign;
                int p, q;
                size_t max_size;
                max_size = nexcit*nexcit*ndets*nwalkers;

                for (int thread = idx; thread < max_size; thread += stride){
                    iex = thread/(ndets*nwalkers*nexcit);
                    
                    jex = thread/(ndets*nwalkers)%nexcit;
                    
                    iw = (thread/ndets)%nwalkers;

                    idet = thread%ndets;
                    
                    p = mapping[cre[idet*nexcit + iex]];
                                
                    q = anh[idet*nexcit + jex];
                    
                    sign = make_cuDoubleComplex(pow(-1.0,(double)(iex + jex)),0.0);
                    
                    mul_tmp[iex*nexcit*nwalkers*ndets + jex*nwalkers*ndets + iw*ndets + idet] = cuCmul(sign, cuCmul(det[iex*nexcit*nwalkers*ndets + jex*nwalkers*ndets + iw*ndets + idet],phases[iw*ndets + idet]));
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + p), mul_tmp[iex*nexcit*nwalkers*ndets + jex*nwalkers*ndets + iw*ndets + idet].x);
                    atomicAdd((double*)(CI+iw*CI_dim2*CI_dim3 + q*CI_dim3 + p)+1, mul_tmp[iex*nexcit*nwalkers*ndets + jex*nwalkers*ndets + iw*ndets + idet].y);
                }
                
            }   
            """,
        "reduce_nfold_cofac_kernel2",
    )

    reduce_nfold_cofac_kernel2(
        (int(numpy.ceil(nexcit * nexcit * ndets * nwalkers / 64)),),
        (64,),
        (
            nexcit,
            cre,
            anh,
            mapping,
            nwalkers,
            ndets,
            CI.shape[1],
            CI.shape[2],
            phases,
            det,
            rhs,
            CI,
        ),
    )


# Energy evaluation


def fill_os_singles_gpu(cre, anh, mapping, offset, chol_factor, spin_buffer_cupy, det_sls):
    """Fill opposite spin (os) contributions from singles.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    ps = cre[:, 0]
    qs = anh[:, 0]
    ndets = ps.shape[0]
    start = det_sls.start

    ps = cupy.asarray(ps)
    qs = cupy.asarray(qs)
    mapping = cupy.asarray(mapping)

    fill_os_singles_kernel = cupy.ElementwiseKernel(
        "raw int32 ps, raw int32 qs, raw complex128 chol_factor, raw int32 mapping, raw int32 ndets, raw int32 nchol, raw int32 nact_shape1, raw int32 nact_shape2",
        "complex128 spin_buffer",
        """
                int q, mappingp;
                q = qs[(i/nchol)%ndets];
                mappingp = mapping[ps[(i/nchol)%ndets]];
                spin_buffer = chol_factor[i/(ndets*nchol)*(nact_shape1*nact_shape2*nchol) + q*nact_shape2*nchol + mappingp*nchol + i%nchol];
                
            """,
        "fill_os_singles_kernel",
    )

    fill_os_singles_kernel(
        ps,
        qs,
        chol_factor,
        mapping,
        ndets,
        chol_factor[0].shape[2],
        chol_factor[0].shape[0],
        chol_factor[0].shape[1],
        spin_buffer_cupy[:, start : start + ndets, :],
    )

    return spin_buffer_cupy


def fill_os_doubles_gpu(cre, anh, mapping, offset, G0, chol_factor, spin_buffer_cupy, det_sls):
    """Fill opposite spin (os) contributions from doubles.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    G0 : np.ndarray
        Half-rotated reference Green's function.
    offset : int
        Offset for frozen core.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    start = det_sls.start
    ndets = cre.shape[0]

    G0_real = G0.real.copy()
    G0_imag = G0.imag.copy()

    fill_os_doubles_kernel = cupy.ElementwiseKernel(
        "raw int32 cre1, raw int32 cre2, raw int32 anh1, raw int32 anh2, raw complex128 chol_factor, raw float64 G0_real, raw float64 G0_imag, int32 offset, raw int32 mapping, int32 ndets, int32 chol_shape, int32 nact_shape1, int32 nact_shape2, int32 Gshape0, int32 Gshape1",
        "complex128 spin_buffer",
        """
                #include <cupy/complex.cuh>
                int p,q,r,s,po,qo,ro,so,iw;
                p = mapping[cre1[(i/chol_shape)%ndets]];
                q = anh1[(i/chol_shape)%ndets];
                r = mapping[cre2[(i/chol_shape)%ndets]];
                s = anh2[(i/chol_shape)%ndets];
                po = cre1[(i/chol_shape)%ndets] + offset;
                qo = anh1[(i/chol_shape)%ndets] + offset;
                ro = cre2[(i/chol_shape)%ndets] + offset;
                so = anh2[(i/chol_shape)%ndets] + offset;
                iw = i/(ndets*chol_shape);
                spin_buffer = (
                    chol_factor[iw*nact_shape1*nact_shape2*chol_shape + q*nact_shape2*chol_shape + p*chol_shape + i%chol_shape]*G0_real[iw*Gshape0*Gshape1 + ro*Gshape1 + so] 
                    + complex<double>(0,1)*(chol_factor[iw*nact_shape1*nact_shape2*chol_shape + q*nact_shape2*chol_shape + p*chol_shape + i%chol_shape]*G0_imag[iw*Gshape0*Gshape1 + ro*Gshape1 + so])                      
                    - 
                    chol_factor[iw*nact_shape1*nact_shape2*chol_shape + s*nact_shape2*chol_shape + p*chol_shape + i%chol_shape]*G0_real[iw*Gshape0*Gshape1 + ro*Gshape1 + qo] 
                    - complex<double>(0,1)*(chol_factor[iw*nact_shape1*nact_shape2*chol_shape + s*nact_shape2*chol_shape + p*chol_shape + i%chol_shape]*G0_imag[iw*Gshape0*Gshape1 + ro*Gshape1 + qo])                    
                    - 
                    chol_factor[iw*nact_shape1*nact_shape2*chol_shape + q*nact_shape2*chol_shape + r*chol_shape + i%chol_shape]*G0_real[iw*Gshape0*Gshape1 + po*Gshape1 + so] 
                    - complex<double>(0,1)*(chol_factor[iw*nact_shape1*nact_shape2*chol_shape + q*nact_shape2*chol_shape + r*chol_shape + i%chol_shape]*G0_imag[iw*Gshape0*Gshape1 + po*Gshape1 + so])
                    + 
                    chol_factor[iw*nact_shape1*nact_shape2*chol_shape + s*nact_shape2*chol_shape + r*chol_shape + i%chol_shape]*G0_real[iw*Gshape0*Gshape1 + po*Gshape1 + qo] 
                    + complex<double>(0,1)*(chol_factor[iw*nact_shape1*nact_shape2*chol_shape + s*nact_shape2*chol_shape + r*chol_shape + i%chol_shape]*G0_imag[iw*Gshape0*Gshape1 + po*Gshape1 + qo])
                    );
                
        """,
        "fill_os_doubles_kernel",
    )

    fill_os_doubles_kernel(
        cre[:, 0],
        cre[:, 1],
        anh[:, 0],
        anh[:, 1],
        chol_factor,
        G0_real,
        G0_imag,
        offset,
        mapping,
        ndets,
        spin_buffer_cupy[0, start, :].shape[0],
        chol_factor[0].shape[0],
        chol_factor[0].shape[1],
        G0_real[0].shape[0],
        G0_real[0].shape[1],
        spin_buffer_cupy[:, start : start + ndets, :],
    )


def fill_os_triples_gpu(cre, anh, mapping, offset, G0, chol_factor, spin_buffer_cupy, det_sls):
    """Fill opposite spin (os) contributions from triples.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        Half-rotated reference Green's function.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    start = det_sls.start
    ndets = cre.shape[0]
    G0_real = G0.real.copy()
    G0_imag = G0.imag.copy()

    fill_os_triples_kernel = cupy.ElementwiseKernel(
        "raw int32 cre0, raw int32 cre1, raw int32 cre2, raw int32 anh0, raw int32 anh1, raw int32 anh2, raw complex128 chol_factor, raw float64 G0_real, raw float64 G0_imag, int32 offset, raw int32 mapping, int32 ndets, int32 chol_shape, int32 nact_shape1, int32 nact_shape2, int32 G0_dim2, int32 G0_dim3",
        "complex128 spin_buffer",
        """
                #include <cupy/complex.cuh>
                int p,q,r,s,t,u,po,qo,ro,so,to,uo,iw,G0_idx_rs, G0_idx_tu, G0_idx_ru, G0_idx_ts, G0_idx_rq, G0_idx_tq, G0_idx_ps, G0_idx_pu, G0_idx_pq;
                p = mapping[cre0[(i/chol_shape)%ndets]];
                q = anh0[(i/chol_shape)%ndets];
                r = mapping[cre1[(i/chol_shape)%ndets]];
                s = anh1[(i/chol_shape)%ndets];
                t = mapping[cre2[(i/chol_shape)%ndets]];
                u = anh2[(i/chol_shape)%ndets];
                po = cre0[(i/chol_shape)%ndets] + offset;
                qo = anh0[(i/chol_shape)%ndets] + offset;
                ro = cre1[(i/chol_shape)%ndets] + offset;
                so = anh1[(i/chol_shape)%ndets] + offset;
                to = cre2[(i/chol_shape)%ndets] + offset;
                uo = anh2[(i/chol_shape)%ndets] + offset;
                iw = i/(ndets*chol_shape);
                
                G0_idx_rs = iw*G0_dim2*G0_dim3 + ro*G0_dim3 + so;
                G0_idx_tu = iw*G0_dim2*G0_dim3 + to*G0_dim3 + uo;
                G0_idx_ru = iw*G0_dim2*G0_dim3 + ro*G0_dim3 + uo;
                G0_idx_ts = iw*G0_dim2*G0_dim3 + to*G0_dim3 + so;
                G0_idx_rq = iw*G0_dim2*G0_dim3 + ro*G0_dim3 + qo;
                G0_idx_tq = iw*G0_dim2*G0_dim3 + to*G0_dim3 + qo;
                G0_idx_ps = iw*G0_dim2*G0_dim3 + po*G0_dim3 + so;
                G0_idx_pu = iw*G0_dim2*G0_dim3 + po*G0_dim3 + uo;
                G0_idx_pq = iw*G0_dim2*G0_dim3 + po*G0_dim3 + qo;
                
                spin_buffer = (
                    chol_factor[iw*nact_shape1*nact_shape2*chol_shape + q*nact_shape2*chol_shape + p*chol_shape + i%chol_shape]*(
                    (G0_real[G0_idx_rs]*G0_real[G0_idx_tu] - G0_real[G0_idx_ru]*G0_real[G0_idx_ts]) 
                    - (G0_imag[G0_idx_rs]*G0_imag[G0_idx_tu] - G0_imag[G0_idx_ru]*G0_imag[G0_idx_ts])
                    + complex<double>(0,1)*((G0_real[G0_idx_rs]*G0_imag[G0_idx_tu] - G0_real[G0_idx_ru]*G0_imag[G0_idx_ts])
                    + (G0_imag[G0_idx_rs]*G0_real[G0_idx_tu] - G0_imag[G0_idx_ru]*G0_real[G0_idx_ts]))
                    )         
                    - chol_factor[iw*nact_shape1*nact_shape2*chol_shape + s*nact_shape2*chol_shape + p*chol_shape + i%chol_shape]*(
                    (G0_real[G0_idx_rq]*G0_real[G0_idx_tu] - G0_real[G0_idx_ru]*G0_real[G0_idx_tq])
                    -(G0_imag[G0_idx_rq]*G0_imag[G0_idx_tu] - G0_imag[G0_idx_ru]*G0_imag[G0_idx_tq])
                    + complex<double>(0,1)*((G0_real[G0_idx_rq]*G0_imag[G0_idx_tu] - G0_real[G0_idx_ru]*G0_imag[G0_idx_tq])
                    +(G0_imag[G0_idx_rq]*G0_real[G0_idx_tu] - G0_imag[G0_idx_ru]*G0_real[G0_idx_tq]))
                    )        
                    + chol_factor[iw*nact_shape1*nact_shape2*chol_shape + u*nact_shape2*chol_shape + p*chol_shape + i%chol_shape]*(
                    (G0_real[G0_idx_rq]*G0_real[G0_idx_ts] - G0_real[G0_idx_rs]*G0_real[G0_idx_tq]) 
                    - (G0_imag[G0_idx_rq]*G0_imag[G0_idx_ts] - G0_imag[G0_idx_rs]*G0_imag[G0_idx_tq])
                    + complex<double>(0,1)*((G0_real[G0_idx_rq]*G0_imag[G0_idx_ts] - G0_real[G0_idx_rs]*G0_imag[G0_idx_tq])
                    + (G0_imag[G0_idx_rq]*G0_real[G0_idx_ts] - G0_imag[G0_idx_rs]*G0_real[G0_idx_tq]))
                    )         
                    - chol_factor[iw*nact_shape1*nact_shape2*chol_shape + q*nact_shape2*chol_shape + r*chol_shape + i%chol_shape]*(
                    (G0_real[G0_idx_ps]*G0_real[G0_idx_tu] - G0_real[G0_idx_ts]*G0_real[G0_idx_pu]) 
                    - (G0_imag[G0_idx_ps]*G0_imag[G0_idx_tu] - G0_imag[G0_idx_ts]*G0_imag[G0_idx_pu])
                    + complex<double>(0,1)*((G0_real[G0_idx_ps]*G0_imag[G0_idx_tu] - G0_real[G0_idx_ts]*G0_imag[G0_idx_pu])
                    + (G0_imag[G0_idx_ps]*G0_real[G0_idx_tu] - G0_imag[G0_idx_ts]*G0_real[G0_idx_pu]))
                    )
                    + chol_factor[iw*nact_shape1*nact_shape2*chol_shape + s*nact_shape2*chol_shape + r*chol_shape + i%chol_shape]*(
                    (G0_real[G0_idx_pq]*G0_real[G0_idx_tu] - G0_real[G0_idx_tq]*G0_real[G0_idx_pu]) 
                    - (G0_imag[G0_idx_pq]*G0_imag[G0_idx_tu] - G0_imag[G0_idx_tq]*G0_imag[G0_idx_pu])
                    + complex<double>(0,1)*((G0_real[G0_idx_pq]*G0_imag[G0_idx_tu] - G0_real[G0_idx_tq]*G0_imag[G0_idx_pu])
                    + (G0_imag[G0_idx_pq]*G0_real[G0_idx_tu] - G0_imag[G0_idx_tq]*G0_real[G0_idx_pu]))
                    )
                    - chol_factor[iw*nact_shape1*nact_shape2*chol_shape + u*nact_shape2*chol_shape + r*chol_shape + i%chol_shape]*(
                    (G0_real[G0_idx_pq]*G0_real[G0_idx_ts] - G0_real[G0_idx_tq]*G0_real[G0_idx_ps]) 
                    - (G0_imag[G0_idx_pq]*G0_imag[G0_idx_ts] - G0_imag[G0_idx_tq]*G0_imag[G0_idx_ps])
                    + complex<double>(0,1)*((G0_real[G0_idx_pq]*G0_imag[G0_idx_ts] - G0_real[G0_idx_tq]*G0_imag[G0_idx_ps])
                    + (G0_imag[G0_idx_pq]*G0_real[G0_idx_ts] - G0_imag[G0_idx_tq]*G0_real[G0_idx_ps]))
                    )
                    + chol_factor[iw*nact_shape1*nact_shape2*chol_shape + q*nact_shape2*chol_shape + t*chol_shape + i%chol_shape]*(
                    (G0_real[G0_idx_ps]*G0_real[G0_idx_ru] - G0_real[G0_idx_rs]*G0_real[G0_idx_pu]) 
                    - (G0_imag[G0_idx_ps]*G0_imag[G0_idx_ru] - G0_imag[G0_idx_rs]*G0_imag[G0_idx_pu])
                    + complex<double>(0,1)*((G0_real[G0_idx_ps]*G0_imag[G0_idx_ru] - G0_real[G0_idx_rs]*G0_imag[G0_idx_pu])
                    + (G0_imag[G0_idx_ps]*G0_real[G0_idx_ru] - G0_imag[G0_idx_rs]*G0_real[G0_idx_pu]))
                    )
                    - chol_factor[iw*nact_shape1*nact_shape2*chol_shape + s*nact_shape2*chol_shape + t*chol_shape + i%chol_shape]*(
                    (G0_real[G0_idx_pq]*G0_real[G0_idx_ru] - G0_real[G0_idx_rq]*G0_real[G0_idx_pu]) 
                    - (G0_imag[G0_idx_pq]*G0_imag[G0_idx_ru] - G0_imag[G0_idx_rq]*G0_imag[G0_idx_pu])
                    + complex<double>(0,1)*((G0_real[G0_idx_pq]*G0_imag[G0_idx_ru] - G0_real[G0_idx_rq]*G0_imag[G0_idx_pu])
                    + (G0_imag[G0_idx_pq]*G0_real[G0_idx_ru] - G0_imag[G0_idx_rq]*G0_real[G0_idx_pu]))
                    )
                    + chol_factor[iw*nact_shape1*nact_shape2*chol_shape + u*nact_shape2*chol_shape + t*chol_shape + i%chol_shape]*(
                    (G0_real[G0_idx_pq]*G0_real[G0_idx_rs] - G0_real[G0_idx_rq]*G0_real[G0_idx_ps])
                    - (G0_imag[G0_idx_pq]*G0_imag[G0_idx_rs] - G0_imag[G0_idx_rq]*G0_imag[G0_idx_ps]) 
                    + complex<double>(0,1)*((G0_real[G0_idx_pq]*G0_imag[G0_idx_rs] - G0_real[G0_idx_rq]*G0_imag[G0_idx_ps])
                    + (G0_imag[G0_idx_pq]*G0_real[G0_idx_rs] - G0_imag[G0_idx_rq]*G0_real[G0_idx_ps]))
                    )
                    )
                    ;
                
        """,
        "fill_os_triples_kernel",
    )

    fill_os_triples_kernel(
        cre[:, 0],
        cre[:, 1],
        cre[:, 2],
        anh[:, 0],
        anh[:, 1],
        anh[:, 2],
        chol_factor,
        G0_real,
        G0_imag,
        offset,
        mapping,
        ndets,
        spin_buffer_cupy[0, start, :].shape[0],
        chol_factor[0].shape[0],
        chol_factor[0].shape[1],
        G0_real[0].shape[1],
        G0_real[0].shape[1],
        spin_buffer_cupy[:, start : start + ndets, :],
    )


def get_ss_doubles_gpu(cre, anh, mapping, chol_fact, buffer, det_sls):
    """Fill same spin (ss) contributions from doubles.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    chol_fact : np.ndarray
        Lxqp intermediate constructed elsewhere.
    buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    start = det_sls.start
    ndets = cre.shape[0]
    nwalkers = chol_fact.shape[0]

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    get_ss_doubles_kernel = cupy.RawKernel(
        r"""
                #include<cuComplex.h>
                extern "C" __global__
                void get_ss_doubles_kernel(int* cre, int* anh, int* mapping, int start, int det_dim, int nwalkers, int ndets, int nact, int nelec, int nchol, cuDoubleComplex* chol_factor, cuDoubleComplex* buffer){
                    int idx = threadIdx.x + blockIdx.x * blockDim.x;
                    int stride = gridDim.x * blockDim.x;
                    int iw, idet, ichol;
                    size_t max_size;
                    int p, q, r, s;
                    max_size = nwalkers*ndets;

                    for (int thread = idx; thread < max_size; thread += stride){
                    
                        iw = thread/ndets;
                        
                        idet =  thread%ndets;

                        p = mapping[cre[idet*2]];
                        q = anh[idet*2];
                        r = mapping[cre[idet*2 + 1]];
                        s = anh[idet*2 + 1];

                        for(ichol = 0; ichol < nchol; ichol++){
                            
                            buffer[iw*det_dim + start + idet] = cuCadd(buffer[iw*det_dim + start + idet], cuCsub(cuCmul(chol_factor[iw*nact*nelec*nchol + q*nelec*nchol + p*nchol + ichol],chol_factor[iw*nact*nelec*nchol + s*nelec*nchol + r*nchol + ichol]), cuCmul(chol_factor[iw*nact*nelec*nchol + q*nelec*nchol + r*nchol + ichol],chol_factor[iw*nact*nelec*nchol + s*nelec*nchol + p*nchol + ichol])));
            
                        }                           
                    }
                    
                }
                """,
        "get_ss_doubles_kernel",
    )
    get_ss_doubles_kernel(
        (int(numpy.ceil(ndets * nwalkers / 64)),),
        (64,),
        (
            cre,
            anh,
            mapping,
            start,
            buffer.shape[1],
            nwalkers,
            ndets,
            chol_fact[0].shape[0],
            chol_fact[0].shape[1],
            chol_fact[0].shape[2],
            chol_fact,
            buffer,
        ),
    )


def build_cofactor_matrix_gpu(row, col, det_matrix, cofactor):
    """Build cofactor matrix with 2 rows/cols deleted.

    Parameters
    ----------
    row_1 : int
        Row to delete when building cofactor.
    col_1 : int
        Column to delete when building cofactor.
    row_2 : int
        Row to delete when building cofactor.
    col_2 : int
        Column to delete when building cofactor.
    det_matrix : np.ndarray
        Precomputed array of determinants <D_I|phi> for given excitation level.
    cofactor : np.ndarray
        Cofactor matrix.

    Returns
    -------
    None
    """
    nwalker = det_matrix.shape[0]
    ndet = det_matrix.shape[1]
    nexcit = det_matrix.shape[2]
    if nexcit - 1 <= 0:
        cofactor[:, :, 0, 0] = cupy.full(cofactor[:, :, 0, 0].shape, 1.0 + 0j)

    build_cofac_Kernel = cupy.RawKernel(
        r"""
                #include<cuComplex.h>
                extern "C" __global__
                void build_cofac_Kernel(int row, int col, int nwalkers, int ndets, int nexcit, cuDoubleComplex* det_matrix, cuDoubleComplex* cofactor){
                    int idx = threadIdx.x + blockIdx.x * blockDim.x;
                    int stride = gridDim.x * blockDim.x;
                    int iw, idet;
                    size_t max_size;
                    int ishift, jshift, ii, jj;
                    max_size = nwalkers*ndets;

                    for (int thread = idx; thread < max_size; thread += stride){
                        
                        iw = thread/ndets;
                        idet = thread%ndets;
                        
                        for (int i=0; i < nexcit; i++){
                            ishift = 0;
                            jshift = 0;
                            if (i > row){
                                ishift = 1;
                            }
                            if ((i == (nexcit - 1)) && (row == (nexcit - 1))){
                                continue;
                            }
                            for (int j=0; j<nexcit; j++){
                                if (j > col){
                                    jshift = 1;
                                }
                                if ((j == (nexcit - 1)) && (col == (nexcit - 1))){
                                    continue;
                                }
                                
                                cofactor[iw*ndets*(nexcit-1)*(nexcit-1) + idet*(nexcit-1)*(nexcit-1) + (i-ishift)*(nexcit-1) + j-jshift] = det_matrix[iw*ndets*nexcit*nexcit + idet*nexcit*nexcit + i*nexcit + j];
                                
                            }
                        }
                    }
                }
                """,
        "build_cofac_Kernel",
    )

    build_cofac_Kernel(
        (int(numpy.ceil(ndet * nwalker / 64)),),
        (64,),
        (row, col, nwalker, ndet, nexcit, det_matrix, cofactor),
    )


def build_cofactor_matrix_4_gpu(row_1, col_1, row_2, col_2, det_matrix, cofactor):
    """Build cofactor matrix with 2 rows/cols deleted.

    Parameters
    ----------
    row_1 : int
        Row to delete when building cofactor.
    col_1 : int
        Column to delete when building cofactor.
    row_2 : int
        Row to delete when building cofactor.
    col_2 : int
        Column to delete when building cofactor.
    det_matrix : np.ndarray
        Precomputed array of determinants <D_I|phi> for given excitation level.
    cofactor : np.ndarray
        Cofactor matrix.

    Returns
    -------
    None
    """
    nwalker = det_matrix.shape[0]
    ndet = det_matrix.shape[1]
    nexcit = det_matrix.shape[2]
    if nexcit - 2 <= 0:
        cofactor[:, :, 0, 0] = cupy.full(cofactor[:, :, 0, 0].shape, 1.0 + 0j)

    build_cofac_4_Kernel = cupy.RawKernel(
        r"""
                #include<cuComplex.h>
                extern "C" __global__
                void build_cofac_4_Kernel(int row_1, int col_1, int row_2, int col_2, int nwalkers, int ndets, int nexcit, cuDoubleComplex* det_matrix, cuDoubleComplex* cofactor){
                    int idx = threadIdx.x + blockIdx.x * blockDim.x;
                    int stride = gridDim.x * blockDim.x;
                    int iw, idet;
                    size_t max_size;
                    int ishift_1, jshift_1, ishift_2, jshift_2, ii, jj;
                    max_size = nwalkers*ndets;

                    for (int thread = idx; thread < max_size; thread += stride){
                        
                        iw = thread/ndets;
                        idet = thread%ndets;
                        
                        for (int i=0; i < nexcit; i++){
                            ishift_1 = 0;
                            jshift_1 = 0;
                            ishift_2 = 0;
                            jshift_2 = 0;
                            if (i > row_1){
                                ishift_1 = 1;
                            }
                            if (i > row_2){
                                ishift_2 = 1;
                            }
                            if ((i == (nexcit - 2)) && (row_1 == (nexcit - 2))){
                                continue;
                            }
                            if ((i == (nexcit - 1)) && (row_2 == (nexcit - 1))){
                                continue;
                            }
                            for (int j=0; j<nexcit; j++){
                                if (j > col_1){
                                    jshift_1 = 1;
                                }
                                if (j > col_2){
                                    jshift_2 = 1;
                                }
                                if ((j == (nexcit - 2)) && (col_1 == (nexcit - 2))){
                                    continue;
                                }
                                if ((j == (nexcit - 1)) && (col_2 == (nexcit - 1))){
                                    continue;
                                }
                                ii = (i - (ishift_1 + ishift_2)>0)?(i - (ishift_1 + ishift_2)):0;
                                jj = (j - (jshift_1 + jshift_2)>0)?(j - (jshift_1 + jshift_2)):0;
                                cofactor[iw*ndets*(nexcit-2)*(nexcit-2) + idet*(nexcit-2)*(nexcit-2) + ii*(nexcit-2) + jj] = det_matrix[iw*ndets*nexcit*nexcit + idet*nexcit*nexcit + i*nexcit + j];
                                
                            }
                        }
                    }
                }
                """,
        "build_cofac_4_Kernel",
    )

    build_cofac_4_Kernel(
        (int(numpy.ceil(ndet * nwalker / 64)),),
        (64,),
        (row_1, col_1, row_2, col_2, nwalker, ndet, nexcit, det_matrix, cofactor),
    )


def reduce_os_spin_factor_gpu(
    ps, qs, mapping, phase, det_cofactor, chol_factor, spin_buffer_cupy, det_sls
):
    """Reduce opposite spin (os) contributions into spin_buffer.

    Parameters
    ----------
    ps : np.ndarray
        Array containing orbitals excitations of occupied.
    qs : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    phases : np.ndarray
        Phase factors.
    cof_mat: np.ndarray
        Cofactor matrix previously constructed.
    chol_fact : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """

    ps = cupy.asarray(ps)
    qs = cupy.asarray(qs)
    mapping = cupy.asarray(mapping)

    ndets = det_cofactor.shape[1]
    start = det_sls.start

    det_cofactor = phase * det_cofactor

    det_cofactor_real = det_cofactor.real.copy()
    det_cofactor_imag = det_cofactor.imag.copy()
    reduce_os_spinfac = cupy.ElementwiseKernel(
        "raw int32 ps, raw int32 qs, raw complex128 chol_factor, raw int32 mapping, raw int32 ndets, raw int32 chol_shape, raw int32 nact_shape1, raw int32 nact_shape2, raw float64 det_cofactor_real, raw float64 det_cofactor_imag, complex128 spin_buff",
        "complex128 spin_buffer",
        """
            int p;
            p = mapping[ps[(i/chol_shape)%ndets]];
            spin_buffer = chol_factor[i/(ndets*chol_shape)*nact_shape1*nact_shape2*chol_shape + qs[(i/chol_shape)%ndets]*nact_shape2*chol_shape + p*chol_shape + i%chol_shape]*det_cofactor_real[i/(ndets*chol_shape)*ndets + (i/chol_shape)%ndets] 
                + complex<double>(0,1)*(chol_factor[i/(ndets*chol_shape)*nact_shape1*nact_shape2*chol_shape + qs[(i/chol_shape)%ndets]*nact_shape2*chol_shape + p*chol_shape + i%chol_shape]*det_cofactor_imag[i/(ndets*chol_shape)*ndets + (i/chol_shape)%ndets])                
        """,
        "reduce_os_spinfac",
    )

    spin_buffer_cupy[:, start : start + ndets, :] += reduce_os_spinfac(
        ps,
        qs,
        chol_factor,
        mapping,
        ndets,
        chol_factor[0].shape[2],
        chol_factor[0].shape[0],
        chol_factor[0].shape[1],
        det_cofactor_real,
        det_cofactor_imag,
        spin_buffer_cupy[:, start : start + ndets, :],
    )


def fill_os_nfold_gpu(cre, anh, mapping, det_matrix, cof_mat, chol_factor, spin_buffer, det_sls):
    """Fill opposite spin (os) n-fold contributions into spin_buffer.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    det_matrix : np.ndarray
        Array of determinants <D_I|phi> for n-fold excitation.
    cof_mat: np.ndarray
        Cofactor matrix buffer.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    spin_buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """
    nexcit = det_matrix.shape[-1]

    cof_mat_all = cupy.zeros(
        (nexcit, nexcit, cof_mat.shape[0], cof_mat.shape[1], cof_mat.shape[2], cof_mat.shape[3]),
        dtype=numpy.complex128,
    )

    for iex in range(nexcit):
        for jex in range(nexcit):
            build_cofactor_matrix_gpu(iex, jex, det_matrix, cof_mat)
            cof_mat_all[iex, jex, :, :, :, :] = cupy.asarray(cof_mat)

    det_cofactor = cupy.linalg.det(cof_mat_all)

    cof_mat_all = None

    for iex in range(nexcit):
        ps = cre[:, iex]
        for jex in range(nexcit):
            qs = anh[:, jex]

            phase = (-1.0 + 0.0j) ** (iex + jex)
            reduce_os_spin_factor_gpu(
                ps, qs, mapping, phase, det_cofactor[iex, jex], chol_factor, spin_buffer, det_sls
            )

    det_cofactor = None


### using cooperative groups to make reduction more efficient


def get_ss_nfold_gpu(cre, anh, mapping, dets_mat, cof_mat, chol_factor, buffer, det_sls):
    """Build same-spin (ss) n-fold contributions.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    det_matrix : np.ndarray
        Output array of determinants <D_I|phi>.
    cof_mat: np.ndarray
        Cofactor matrix buffer.
    chol_factor : np.ndarray
        Lxqp intermediate constructed elsewhere.
    buffer : np.ndarray
        Buffer for holding contribution.
    det_sls : np.ndarray
        Index slice for this exctitation level's contributions.

    Returns
    -------
    None
    """

    nwalkers = dets_mat.shape[0]
    nexcit = dets_mat.shape[-1]
    ndets = cof_mat.shape[1]
    start = det_sls.start

    cre = cupy.asarray(cre)
    anh = cupy.asarray(anh)
    mapping = cupy.asarray(mapping)

    cof_mat_all = cupy.zeros(
        (
            nexcit,
            nexcit,
            nexcit,
            nexcit,
            cof_mat.shape[0],
            cof_mat.shape[1],
            cof_mat.shape[2],
            cof_mat.shape[3],
        ),
        dtype=numpy.complex128,
    )

    for iex in range(nexcit):
        for jex in range(nexcit):
            for kex in range(iex + 1, nexcit):
                for lex in range(jex + 1, nexcit):
                    build_cofactor_matrix_4_gpu(iex, jex, kex, lex, dets_mat, cof_mat)
                    cof_mat_all[iex, jex, kex, lex, :, :, :, :] = cof_mat

    det_cofactor = cupy.linalg.det(cof_mat_all)

    buffer_copy = cupy.zeros_like(buffer[:, start : start + ndets])

    get_ss_nfold_parallel_kernel = cupy.RawKernel(
        r"""
                #include <cuComplex.h> 
                #include <cooperative_groups.h>
                #include <cooperative_groups/reduce.h>
                
                using namespace cooperative_groups;
                namespace cg = cooperative_groups;
                
                extern "C" __global__
                void get_ss_nfold_parallel_kernel(int nexcit, int* cre, int* anh, int* mapping, int start, int nwalkers, int ndets, int nact, int nelec, int nchol, cuDoubleComplex* chol_factor, cuDoubleComplex* det_cofac, cuDoubleComplex* spin_buffer){

                    int iex, jex, kex, lex, iw, idet, ichol;
                    cuDoubleComplex phase;
                    cuDoubleComplex cholsum_tmp;
                    //cholsum_tmp = make_cuDoubleComplex(0.0,0.0);
                    size_t max_size;
                    
                    auto grid = cg::this_grid();
                    auto block = cg::this_thread_block();
                    auto active = cg::tiled_partition<32>(block);
                    
                    int warp_id = grid.thread_rank() / 32;
                    int stride = (grid.size() + 31) / 32;
                   
                    int p, q, r, s;                   
                    
                    max_size = ndets*nwalkers*nexcit*nexcit*nexcit*nexcit;

                    for (; warp_id < max_size; warp_id += stride){                        
                        
                        iex = warp_id/(ndets*nwalkers*nexcit*nexcit*nexcit);
                        
                        jex = warp_id/(ndets*nwalkers*nexcit*nexcit)%nexcit;
                        
                        iw = (warp_id/ndets)%nwalkers;
                        
                        idet = warp_id%ndets;
                        
                                                        
                        p = mapping[cre[idet*nexcit + iex]];
                                
                        q = anh[idet*nexcit + jex];
                        
                        kex = warp_id/(ndets*nwalkers*nexcit)%nexcit;
                        
                        if (kex<iex+1) {
                            continue;
                        }
                                                                        
                        r = mapping[cre[idet*nexcit + kex]];                        
                            
                        lex = warp_id/(ndets*nwalkers)%nexcit;
                        
                        if (lex<jex+1) {
                            continue;
                        }        
                                
                        s = anh[idet*nexcit + lex];
                                
                        phase = make_cuDoubleComplex(pow(-1.0,(double)(kex + lex + iex + jex)),0.0);
                        
                        cholsum_tmp = make_cuDoubleComplex(0.0,0.0);
                                                                                                                                           
                        for(ichol=active.thread_rank(); ichol < nchol; ichol+=active.size()){
                                    
                                    cholsum_tmp = cuCadd(cholsum_tmp, 
                                    cuCsub(cuCmul(chol_factor[iw*nact*nelec*nchol + s*nelec*nchol + r*nchol + ichol],chol_factor[iw*nact*nelec*nchol + q*nelec*nchol + p*nchol + ichol]), cuCmul(chol_factor[iw*nact*nelec*nchol + q*nelec*nchol + r*nchol + ichol],chol_factor[iw*nact*nelec*nchol + s*nelec*nchol + p*nchol + ichol]))); 
                                    
                        }
                        
                        cholsum_tmp.x = cg::reduce(active, cholsum_tmp.x, cg::plus<double>());
                        cholsum_tmp.y = cg::reduce(active, cholsum_tmp.y, cg::plus<double>());
                        
                        
                        if (active.thread_rank() == 0) {
                            atomicAdd((double*)(spin_buffer + iw*ndets + idet), cuCmul(phase, cuCmul(cholsum_tmp, det_cofac[iex*nexcit*nexcit*nexcit*nwalkers*ndets + jex*nexcit*nexcit*nwalkers*ndets + kex*nexcit*nwalkers*ndets + lex*nwalkers*ndets +iw*ndets + idet])).x);
                            atomicAdd((double*)(spin_buffer + iw*ndets + idet)+1, cuCmul(phase, cuCmul(cholsum_tmp, det_cofac[iex*nexcit*nexcit*nexcit*nwalkers*ndets + jex*nexcit*nexcit*nwalkers*ndets + kex*nexcit*nwalkers*ndets + lex*nwalkers*ndets +iw*ndets + idet])).y);
                        }
                           
                        
                    }
                }
                """,
        "get_ss_nfold_parallel_kernel",
        options=("-std=c++11",),
        backend="nvcc",
    )

    get_ss_nfold_parallel_kernel(
        (int(numpy.ceil(nexcit * nexcit * nexcit * nexcit * ndets * nwalkers / 64)),),
        (64,),
        (
            nexcit,
            cre,
            anh,
            mapping,
            start,
            nwalkers,
            ndets,
            chol_factor[0].shape[0],
            chol_factor[0].shape[1],
            chol_factor[0].shape[2],
            chol_factor,
            det_cofactor,
            buffer_copy,
        ),
    )

    buffer[:, start : start + ndets] = buffer_copy

    det_cofactor = None
