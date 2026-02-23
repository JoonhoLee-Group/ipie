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
# Author: Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#

try:
    # pylint: disable=import-error
    import cupy as cp
    import numba
    from numba import cuda, complex128
except ModuleNotFoundError:
    pass

_block_size = 512
BM = 32
BN = 32
BK = 8
TM = 4
TN = 4

kernel_code_t1 = r"""
#define BM 32
#define BN 32
#define BK 8
#define TM 4
#define TN 4

#include <cuComplex.h>

extern "C" __global__
void get_T1(int naux, int nwalker, int nocc, int nbasis, int nk, int nkcube, const int *kcubelist, const int* kpq_mat, const cuDoubleComplex *A,
                                     const cuDoubleComplex *B, cuDoubleComplex *C) {
    const int M = nocc * naux;
    const int K = nbasis;
    const int N = nocc * nwalker;
  const int batchid = blockIdx.z;
  const unsigned int cRow = blockIdx.y;
  const unsigned int cCol = blockIdx.x;
 //if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0 &&
        //blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //printf("batchid: %d, cRow: %d, cCol: %d\n", batchid, cRow, cCol);

        //for (int i = 0; i < nkcube; ++i) {
           // printf("kcubelist[%d]: %d\n", i, kcubelist[i]);
        //}
   // }
  int ikcube_real = kcubelist[batchid];
    // expand ikcube to q, k, k'
  int iq = ikcube_real / (nk * nk);
  int ik = (ikcube_real % (nk * nk)) / nk;
  int ik_pr = (ikcube_real % (nk * nk)) % nk;
  int ikpq = kpq_mat[iq * nk + ik];
  int ikpr_pq = kpq_mat[iq * nk + ik_pr];

  const unsigned int totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const unsigned int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // BN/TN are the number of threads to span a column
  const unsigned int threadCol = threadIdx.x % (BN / TN);
  const unsigned int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in shared memory
  __shared__ cuDoubleComplex As[BM * BK];
  __shared__ cuDoubleComplex Bs[BK * BN];

  // calculating the indices that this thread will load into shared memory
  const unsigned int innerRowA = threadIdx.x / BK;
  const unsigned int innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const unsigned int strideA = numThreadsBlocktile / BK;
  const unsigned int innerRowB = threadIdx.x / BN;
  const unsigned int innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better global memory coalescing
  const unsigned int strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in register file
  cuDoubleComplex threadResults[TM * TN];
  // Initialize threadResults to zero
  for (int i = 0; i < TM * TN; ++i) {
    threadResults[i] = make_cuDoubleComplex(0.0, 0.0);
  }
  // register caches for As and Bs
  cuDoubleComplex regM[TM];
  cuDoubleComplex regN[TN];

  const cuDoubleComplex* A0 = A;
  const cuDoubleComplex* B0 = B;

  // outer-most loop over block tiles
  for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the shared memory caches
    for (unsigned int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      unsigned int sharedRow = innerRowA + loadOffset;
      unsigned int sharedCol = innerColA;
      unsigned int globalRowA = cRow * BM + sharedRow;
      unsigned int globalColA = bkIdx + sharedCol;
      if (globalRowA < M && globalColA < K) {
        unsigned int a = globalRowA / naux;
        unsigned int X = globalRowA % naux;
        As[sharedRow * BK + sharedCol] = A0[iq * nk * nocc * naux * nbasis + ik * nocc * naux * nbasis + a * naux * nbasis + X * nbasis + globalColA];
      } // A[batchid, globalRowA, globalColA] = rchol[iq, ik, a, X, globalColA], globalRowA = a * naux + X
      else {
        As[sharedRow * BK + sharedCol] = make_cuDoubleComplex(0.0, 0.0);
      }
    }
    for (unsigned int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      unsigned int sharedRow = innerRowB + loadOffset;
      unsigned int sharedCol = innerColB;
      unsigned int globalRowB = bkIdx + sharedRow;
      unsigned int globalColB = cCol * BN + sharedCol;
      if (globalRowB < K && globalColB < N) {
        unsigned int b = globalColB % nocc;
        unsigned int w = globalColB / nocc;
        Bs[sharedRow * BN + sharedCol] = B0[ikpr_pq * nk * nocc * nwalker * nbasis + ikpq * nocc * nwalker * nbasis + w * nocc * nbasis + b * nbasis + globalRowB];
      } // B[batchid, globalRowB, globalColB] = Ghalf[ikpr_pq, ikpq, w, b, globalRowB], globalColB = w * nocc + b 
      else {
        Bs[sharedRow * BN + sharedCol] = make_cuDoubleComplex(0.0, 0.0);
      }
    }
    __syncthreads();

    // calculate per-thread results
    for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (unsigned int i = 0; i < TM; ++i) {
        unsigned int row = threadRow * TM + i;
        regM[i] = As[row * BK + dotIdx];
      }
      for (unsigned int i = 0; i < TN; ++i) {
        unsigned int col = threadCol * TN + i;
        regN[i] = Bs[dotIdx * BN + col];
      }
      for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] = cuCadd(
              threadResults[resIdxM * TN + resIdxN],
              cuCmul(regM[resIdxM], regN[resIdxN]));
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
    unsigned int globalRowC = cRow * BM + threadRow * TM + resIdxM;
    for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
      unsigned int globalColC = cCol * BN + threadCol * TN + resIdxN;
        unsigned int a = globalRowC / naux;
        unsigned int X = globalRowC % naux;
        unsigned int b = globalColC % nocc;
        unsigned int w = globalColC / nocc;
      if (globalRowC < M && globalColC < N) {
      // C[batchid * M * N + a * N * naux + X * N + w * nocc + b] = threadResults[resIdxM * TN + resIdxN];
        C[w * nkcube * nocc * nocc * naux + batchid * nocc * nocc * naux + a * nocc * naux + X * nocc + b] = threadResults[resIdxM * TN + resIdxN];
        // C[batchid, globalRowC, globalColC] = T1[batchid, a, X, w, b] stored as T1[w, batchid*a*X*b]
      }
    }
  }
}
"""

kernel_code_t2 = r"""
#define BM 32
#define BN 32
#define BK 8
#define TM 4
#define TN 4

#include <cuComplex.h>

extern "C" __global__
void get_T2(int naux, int nwalker, int nocc, int nbasis, int nk, int nkcube, const int *kcubelist, const int* kpq_mat, const cuDoubleComplex *A,
                                     const cuDoubleComplex *B, cuDoubleComplex *C) {
    const int M = nocc * nwalker;
    const int K = nbasis;
    const int N = nocc * naux;
  const int batchid = blockIdx.z;
  const unsigned int cRow = blockIdx.y;
  const unsigned int cCol = blockIdx.x;
  int ikcube_real = kcubelist[batchid];
    // expand ikcube to q, k, k'
  int iq = ikcube_real / (nk * nk);
  int ik = (ikcube_real % (nk * nk)) / nk;
  int ik_pr = (ikcube_real % (nk * nk)) % nk;

  const unsigned int totalResultsBlocktile = BM * BN;
  // A thread is responsible for calculating TM*TN elements in the blocktile
  const unsigned int numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

  // BN/TN are the number of threads to span a column
  const unsigned int threadCol = threadIdx.x % (BN / TN);
  const unsigned int threadRow = threadIdx.x / (BN / TN);

  // allocate space for the current blocktile in shared memory
  __shared__ cuDoubleComplex As[BM * BK];
  __shared__ cuDoubleComplex Bs[BK * BN];

  // calculating the indices that this thread will load into shared memory
  const unsigned int innerRowA = threadIdx.x / BK;
  const unsigned int innerColA = threadIdx.x % BK;
  // calculates the number of rows of As that are being loaded in a single step
  // by a single block
  const unsigned int strideA = numThreadsBlocktile / BK;
  const unsigned int innerRowB = threadIdx.x / BN;
  const unsigned int innerColB = threadIdx.x % BN;
  // for both As and Bs we want each load to span the full column-width, for
  // better global memory coalescing
  const unsigned int strideB = numThreadsBlocktile / BN;

  // allocate thread-local cache for results in register file
  cuDoubleComplex threadResults[TM * TN];
  // Initialize threadResults to zero
  for (int i = 0; i < TM * TN; ++i) {
    threadResults[i] = make_cuDoubleComplex(0.0, 0.0);
  }
  // register caches for As and Bs
  cuDoubleComplex regM[TM];
  cuDoubleComplex regN[TN];

  const cuDoubleComplex* A0 = A;
  const cuDoubleComplex* B0 = B;

  // outer-most loop over block tiles
  for (unsigned int bkIdx = 0; bkIdx < K; bkIdx += BK) {
    // populate the shared memory caches
    for (unsigned int loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
      unsigned int sharedRow = innerRowA + loadOffset;
      unsigned int sharedCol = innerColA;
      unsigned int globalRowA = cRow * BM + sharedRow;
      unsigned int globalColA = bkIdx + sharedCol;
      if (globalRowA < M && globalColA < K) {
        unsigned int w = globalRowA / nocc;
        unsigned int a = globalRowA % nocc;
        As[sharedRow * BK + sharedCol] = A0[ik * nk * nocc * nwalker * nbasis + ik_pr * nocc * nwalker * nbasis + w * nocc * nbasis + a * nbasis + globalColA];
      } // A[batchid, globalRowA, globalColA] = Ghalf[ik, ik_pr, w, a, globalColA], globalRowA = w * nocc + a
      else {
        As[sharedRow * BK + sharedCol] = make_cuDoubleComplex(0.0, 0.0);
      }
    }
    for (unsigned int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      unsigned int sharedRow = innerRowB + loadOffset;
      unsigned int sharedCol = innerColB;
      unsigned int globalRowB = bkIdx + sharedRow;
      unsigned int globalColB = cCol * BN + sharedCol;
      if (globalRowB < K && globalColB < N) {
        unsigned int X = globalColB / nocc;
        unsigned int b = globalColB % nocc;
        Bs[sharedRow * BN + sharedCol] = B0[iq * nk * nocc * naux * nbasis + ik_pr * nocc * naux * nbasis + globalRowB * naux * nocc + X * nocc + b];
      } // B[batchid, globalRowB, globalColB] = rcholbar[iq, ik_pr,  globalRowB, X, b], globalColB = X * nocc + b 
      else {
        Bs[sharedRow * BN + sharedCol] = make_cuDoubleComplex(0.0, 0.0);
      }
    }
    __syncthreads();

    // calculate per-thread results
    for (unsigned int dotIdx = 0; dotIdx < BK; ++dotIdx) {
      // block into registers
      for (unsigned int i = 0; i < TM; ++i) {
        unsigned int row = threadRow * TM + i;
        regM[i] = As[row * BK + dotIdx];
      }
      for (unsigned int i = 0; i < TN; ++i) {
        unsigned int col = threadCol * TN + i;
        regN[i] = Bs[dotIdx * BN + col];
      }
      for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
          threadResults[resIdxM * TN + resIdxN] = cuCadd(
              threadResults[resIdxM * TN + resIdxN],
              cuCmul(regM[resIdxM], regN[resIdxN]));
        }
      }
    }
    __syncthreads();
  }

  // write out the results
  for (unsigned int resIdxM = 0; resIdxM < TM; ++resIdxM) {
    unsigned int globalRowC = cRow * BM + threadRow * TM + resIdxM;
    for (unsigned int resIdxN = 0; resIdxN < TN; ++resIdxN) {
      unsigned int globalColC = cCol * BN + threadCol * TN + resIdxN;
      unsigned int X = globalColC / nocc;
      unsigned int b = globalColC % nocc;
      unsigned int w = globalRowC / nocc;
      unsigned int a = globalRowC % nocc;
      if (globalRowC < M && globalColC < N) {
      // C[batchid * M * N + a * N * naux + X * N + w * nocc + b] = threadResults[resIdxM * TN + resIdxN];
        C[w * nkcube * nocc * nocc * naux + batchid * nocc * nocc * naux + a * nocc * naux + X * nocc + b] = threadResults[resIdxM * TN + resIdxN];
        // C[batchid, globalRowC, globalColC] = T2[batchid, X, b, w, a] stored as T2[w, batchid*a*X*b]
      }
    }
  }
}
"""

get_T1_cupy = cp.RawKernel(kernel_code_t1, "get_T1")
get_T2_cupy = cp.RawKernel(kernel_code_t2, "get_T2")


def exx_kpt_kernel(rchol, rcholbar, Ghalf, kcubelist, kpq_mat):
    """Calculate the exchange energy for each walker."""
    nwalkers = Ghalf.shape[2]
    nocc = rchol.shape[2]
    naux = rchol.shape[3]
    nbasis = rchol.shape[-1]
    nk = Ghalf.shape[0]
    T1 = cp.zeros((nwalkers, len(kcubelist) * nocc * naux * nocc), dtype=cp.complex128)
    T2 = cp.zeros((nwalkers, len(kcubelist) * nocc * naux * nocc), dtype=cp.complex128)

    M = max(nocc * naux, nbasis)
    N = max(nocc * nwalkers, nbasis)
    threadsperblock = (BM * BN // (TM * TN), 1, 1)
    blockspergrid_x = (N + BN - 1) // BN
    blockspergrid_y = (M + BM - 1) // BM
    blockspergrid_z = len(kcubelist)

    kcubelist_cupy = cp.array(kcubelist, dtype=cp.int32)
    kcubelist_cupy = cp.ascontiguousarray(kcubelist_cupy)
    kpq_mat_cupy = cp.array(kpq_mat, dtype=cp.int32)
    kpq_mat_cupy = cp.ascontiguousarray(kpq_mat_cupy)

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    args1 = (
        naux,
        nwalkers,
        nocc,
        nbasis,
        nk,
        len(kcubelist),
        kcubelist_cupy,
        kpq_mat_cupy,
        rchol,
        Ghalf,
        T1,
    )

    blockspergrid_x2 = (M + BN - 1) // BN
    blockspergrid_y2 = (N + BM - 1) // BM
    blockspergrid2 = (blockspergrid_x2, blockspergrid_y2, blockspergrid_z)
    args2 = (
        naux,
        nwalkers,
        nocc,
        nbasis,
        nk,
        len(kcubelist),
        kcubelist_cupy,
        kpq_mat_cupy,
        Ghalf,
        rcholbar,
        T2,
    )

    get_T1_cupy(blockspergrid, threadsperblock, args1)
    get_T2_cupy(blockspergrid2, threadsperblock, args2)

    # exx_w = cp.sum(T1 * T2, axis=1)
    exx_w = cp.einsum("wI, wI -> w", T1, T2, optimize=True)
    cp.cuda.stream.get_current_stream().synchronize()
    return exx_w
