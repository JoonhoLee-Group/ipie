try:
    import cupy as cp
except ImportError:
    cp = None

BM = 32
BN = 32
BK = 8
TM = 4
TN = 4

kernel_code_vhs1 = r"""
#define BM 32
#define BN 32
#define BK 8
#define TM 4
#define TN 4

#include <cuComplex.h>

extern "C" __global__
void VHS_construction1(int nq, int nk, int naux, int nbasis, int nwalker, const int *kpq_mat, const cuDoubleComplex *A,
                                     const cuDoubleComplex *B, cuDoubleComplex *C) {
  const int batchid = blockIdx.z;
  const unsigned int cRow = blockIdx.y;
  const unsigned int cCol = blockIdx.x;
  const int M = nwalker;
  const int N = nbasis * nbasis;
  const int K = naux;
  int iq = batchid / nk;
  int ik = batchid % nk;
  int ikpq = kpq_mat[iq * nk + ik];

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
        As[sharedRow * BK + sharedCol] = A0[globalRowA * K * nq + globalColA * nq + iq];
    // xshifted[arow, acol, iq]
      } else {
        As[sharedRow * BK + sharedCol] = make_cuDoubleComplex(0.0, 0.0);
      }
    }
    for (unsigned int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      unsigned int sharedRow = innerRowB + loadOffset;
      unsigned int sharedCol = innerColB;
      unsigned int globalRowB = bkIdx + sharedRow;
      unsigned int globalColB = cCol * BN + sharedCol;
      unsigned int p = globalColB / nbasis;
      unsigned int r = globalColB % nbasis;
      if (globalRowB < K && globalColB < N) {
        Bs[sharedRow * BN + sharedCol] = B0[globalRowB * nk * nq * nbasis * nbasis + ik * nq * nbasis * nbasis + p * nq * nbasis + iq * nbasis + r];
    //chol[brow, ik, p, iq, r]
      } else {
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
    unsigned int p = globalColC / nbasis;
    unsigned int r = globalColC % nbasis;
      if (globalRowC < M && globalColC < N) {
        C[globalRowC * nk * nk * nbasis * nbasis + ik * nbasis * nk * nbasis + p * nbasis * nk + ikpq * nbasis + r] =
            cuCadd(C[globalRowC * nk * nk * nbasis * nbasis + ik * nbasis * nk * nbasis + p * nbasis * nk + ikpq * nbasis + r], threadResults[resIdxM * TN + resIdxN]);
      }
    }
  }
}
"""

kernel_code_vhs2 = r"""
#define BM 32
#define BN 32
#define BK 8
#define TM 4
#define TN 4

#include <cuComplex.h>

extern "C" __global__
void VHS_construction2(int nq, int nk, int naux, int nbasis, int nwalker, const int *kpq_mat, const cuDoubleComplex *A,
                                     const cuDoubleComplex *B, cuDoubleComplex *C) {
  const int batchid = blockIdx.z;
  const unsigned int cRow = blockIdx.y;
  const unsigned int cCol = blockIdx.x;
  const int M = nwalker;
  const int N = nbasis * nbasis;
  const int K = naux;
  int iq = batchid / nk;
  int ik = batchid % nk;
  int ikpq = kpq_mat[iq * nk + ik];

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
        As[sharedRow * BK + sharedCol] = A0[globalRowA * K * nq + globalColA * nq + iq];
    // xshifted[arow, acol, iq]
      } else {
        As[sharedRow * BK + sharedCol] = make_cuDoubleComplex(0.0, 0.0);
      }
    }
    for (unsigned int loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
      unsigned int sharedRow = innerRowB + loadOffset;
      unsigned int sharedCol = innerColB;
      unsigned int globalRowB = bkIdx + sharedRow;
      unsigned int globalColB = cCol * BN + sharedCol;
      unsigned int p = globalColB / nbasis;
      unsigned int r = globalColB % nbasis;
      if (globalRowB < K && globalColB < N) {
        Bs[sharedRow * BN + sharedCol] = B0[globalRowB * nk * nq * nbasis * nbasis + ik * nq * nbasis * nbasis + p * nq * nbasis + iq * nbasis + r];
    //chol[brow, ik, p, iq, r]
      } else {
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
        regN[i] = cuConj(Bs[dotIdx * BN + col]);
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
    unsigned int p = globalColC / nbasis;
    unsigned int r = globalColC % nbasis;
      if (globalRowC < M && globalColC < N) {
        C[globalRowC * nk * nk * nbasis * nbasis + ikpq * nbasis * nk * nbasis + r * nbasis * nk + ik * nbasis + p] =
            cuCadd(C[globalRowC * nk * nk * nbasis * nbasis + ikpq * nbasis * nk * nbasis + r * nbasis * nk + ik * nbasis + p], threadResults[resIdxM * TN + resIdxN]);
      }
    }
  }
}
"""

kernel_VHS1_cupy = cp.RawKernel(kernel_code_vhs1, "VHS_construction1")
kernel_VHS2_cupy = cp.RawKernel(kernel_code_vhs2, "VHS_construction2")


def call_kernel_VHS_construction1(chol, xshifted, naux, nk, nbasis, nwalker, ikpq_mat, VHS):
    ikpq_mat = cp.asarray(ikpq_mat, dtype=cp.int32)
    M = max(nwalker, naux)
    N = max(naux, nbasis**2)
    nq = xshifted.shape[-1]
    blockspergrid_x = (N + BN - 1) // BN
    blockspergrid_y = (M + BM - 1) // BM
    blockspergrid_z = ikpq_mat.shape[0] * nk

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    threadsperblock = ((BN * BM) // (TN * TM), 1, 1)
    args = (nq, nk, naux, nbasis, nwalker, ikpq_mat, xshifted, chol, VHS)
    kernel_VHS1_cupy(blockspergrid, threadsperblock, args)
    cp.cuda.stream.get_current_stream().synchronize()


def call_kernel_VHS_construction2(chol, xshifted, naux, nk, nbasis, nwalker, ikpq_mat, VHS):
    ikpq_mat = cp.asarray(ikpq_mat, dtype=cp.int32)
    M = max(nwalker, naux)
    N = max(naux, nbasis**2)
    nq = xshifted.shape[-1]
    blockspergrid_x = (N + BN - 1) // BN
    blockspergrid_y = (M + BM - 1) // BM
    blockspergrid_z = ikpq_mat.shape[0] * nk

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    threadsperblock = ((BN * BM) // (TN * TM), 1, 1)
    args = (nq, nk, naux, nbasis, nwalker, ikpq_mat, xshifted, chol, VHS)
    kernel_VHS2_cupy(blockspergrid, threadsperblock, args)
    cp.cuda.stream.get_current_stream().synchronize()
