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

_block_size = 512  # can be optimized
BM = 32
BN = 64
BK = 8
TM = 4
TN = 4

@cuda.jit("void(complex128[:, :, :, :, :], complex128[:, :, :, :, :], int64[:], int64[:, :], complex128[:, :, :, :, :])")
# rchol[q, k, i, X, p], rcholbar[q, k, p, X, i], Ghalf[k1, k2, w, i, p], kpq_mat
def get_T1(rchol, Ghalf, kcubelist, kpq_mat, T1):

    # kcubelist: batchd index I
    # M: nocc * naux
    # K: nbasis
    # N: nwalkers * nocc
    
    nq = rchol.shape[0]
    nk = rchol.shape[1]
    naux = rchol.shape[3]
    nocc = rchol.shape[2]
    nbasis = rchol.shape[-1]
    nwalker = Ghalf.shape[2]
    M = naux * nocc
    K = nbasis
    N = nocc * nwalker
    nkcube = len(kcubelist)

    batch_id = cuda.blockIdx.z # I
    cRow = cuda.blockIdx.y
    cCol = cuda.blockIdx.x
    threadCol = cuda.threadIdx.x
    threadRow = cuda.threadIdx.y
    threadsPerRow = BN // TN
    thread_id = threadRow * threadsPerRow + threadCol

    ikcube_real = kcubelist[batch_id]
    # expand ikcube to q, k, k'
    iq = ikcube_real // (nk * nk)
    ik = (ikcube_real % (nk * nk)) // nk
    ik_pr = (ikcube_real % (nk * nk)) % nk
    ikpq = kpq_mat[iq, ik]
    ikpr_pq = kpq_mat[iq, ik_pr]
    
   
     # Allocate shared memory for tiles of chol and xshifted
    schol = cuda.shared.array(shape=(BM, BK), dtype=complex128)
    sG1 = cuda.shared.array(shape=(BK, BN), dtype=complex128)
    # Allocate thread-local cache for results
    threadResults = cuda.local.array(shape=(TM, TN), dtype=complex128)
    regM = cuda.local.array(shape=(TM,), dtype=complex128)
    regN = cuda.local.array(shape=(TN,), dtype=complex128)
    # Compute the starting positions
    a_start_row = cRow * BM
    b_start_col = cCol * BN

    # Total number of threads per block
    numThreadsBlocktile = (BM * BN) // (TM * TN)

    # Compute strides
    strideA = (BM * BK + numThreadsBlocktile - 1) // numThreadsBlocktile
    strideB = (BK * BN + numThreadsBlocktile - 1) // numThreadsBlocktile

    # Main loop over the K dimension
    for bkIdx in range(0, K, BK):
        # Load tiles from A into shared memory
        for idx in range(strideA):
            index = thread_id * strideA + idx
            if index < BM * BK:
                row = index // BK
                col = index % BK
                a_row = a_start_row + row
                a_col = bkIdx + col
                a = a_row // naux
                X = a_row % naux
                if a_row < M and a_col < K:
                    schol[row, col] = rchol[iq, ik, a, X, a_col]
                else:
                    schol[row, col] = 0.0

        # Load tiles from B into shared memory
        for idx in range(strideB):
            index = thread_id * strideB + idx
            if index < BK * BN:
                row = index // BN
                col = index % BN
                b_row = bkIdx + row
                b_col = b_start_col + col
                w = b_col // nocc
                b = b_col % nocc
                if b_row < K and b_col < N:
                    sG1[row, col] = Ghalf[ikpr_pq, ikpq, w, b, b_row]
                else:
                    sG1[row, col] = 0.0

        # Synchronize threads after loading
        cuda.syncthreads()

        # Compute per-thread results
        for dotIdx in range(BK):
            # Load elements from shared memory into registers
            for i in range(TM):
                regM[i] = schol[threadRow * TM + i, dotIdx]
            for j in range(TN):
                regN[j] = sG1[dotIdx, threadCol * TN + j]

            # Compute the dot product
            for resIdxM in range(TM):
                for resIdxN in range(TN):
                    threadResults[resIdxM, resIdxN] += regM[resIdxM] * regN[resIdxN]

        # Synchronize threads before the next iteration
        cuda.syncthreads()

    # Write the results back to the global memory matrix C
    for resIdxM in range(TM):
        c_row = a_start_row + threadRow * TM + resIdxM
        for resIdxN in range(TN):
            c_col = b_start_col + threadCol * TN + resIdxN
            a = c_row // naux
            X = c_row % naux
            b = c_col % nocc
            w = c_col // nocc
            if c_row < M and c_col < N:
                T1[batch_id, a, X, w, b] += threadResults[resIdxM, resIdxN]


@cuda.jit("void(complex128[:, :, :, :, :], complex128[:, :, :, :, :], int64[:], int64[:, :], complex128[:, :, :, :, :])")
# rchol[q, k, i, X, p], rcholbar[q, k, p, X, i], Ghalf[k1, k2, w, i, p], kpq_mat
def get_T2(rcholbar, Ghalf, kcubelist, kpq_mat, T2):

    # kcubelist: batchd index I
    # M: nwalkers * nocc
    # K: nbasis
    # N: naux * nocc
    
    nq = rcholbar.shape[0]
    nk = rcholbar.shape[1]
    naux = rcholbar.shape[3]
    nocc = rcholbar.shape[-1]
    nbasis = rcholbar.shape[2]
    nwalker = Ghalf.shape[2]
    M = nwalker * nocc
    K = nbasis
    N = nocc * naux
    nkcube = len(kcubelist)

    batch_id = cuda.blockIdx.z # I
    cRow = cuda.blockIdx.y
    cCol = cuda.blockIdx.x
    threadCol = cuda.threadIdx.x
    threadRow = cuda.threadIdx.y
    threadsPerRow = BN // TN
    thread_id = threadRow * threadsPerRow + threadCol

    ikcube_real = kcubelist[batch_id]
    # expand ikcube to q, k, k'
    iq = ikcube_real // (nk * nk)
    ik = (ikcube_real % (nk * nk)) // nk
    ik_pr = (ikcube_real % (nk * nk)) % nk
    ikpq = kpq_mat[iq, ik]
    ikpr_pq = kpq_mat[iq, ik_pr]
    
   
     # Allocate shared memory for tiles of chol and xshifted
    scholbar = cuda.shared.array(shape=(BK, BN), dtype=complex128)
    sG2 = cuda.shared.array(shape=(BM, BK), dtype=complex128)
    # Allocate thread-local cache for results
    threadResults = cuda.local.array(shape=(TM, TN), dtype=complex128)
    regM = cuda.local.array(shape=(TM,), dtype=complex128)
    regN = cuda.local.array(shape=(TN,), dtype=complex128)
    # Compute the starting positions
    a_start_row = cRow * BM
    b_start_col = cCol * BN

    # Total number of threads per block
    numThreadsBlocktile = (BM * BN) // (TM * TN)

    # Compute strides
    strideA = (BM * BK + numThreadsBlocktile - 1) // numThreadsBlocktile
    strideB = (BK * BN + numThreadsBlocktile - 1) // numThreadsBlocktile

    # Main loop over the K dimension
    for bkIdx in range(0, K, BK):
        # Load tiles from A into shared memory
        for idx in range(strideA):
            index = thread_id * strideA + idx
            if index < BM * BK:
                row = index // BK
                col = index % BK
                a_row = a_start_row + row
                a_col = bkIdx + col
                w = a_row // nocc
                a = a_row % nocc
                if a_row < M and a_col < K:
                    sG2[row, col] = Ghalf[ik, ik_pr, w, a, a_col]
                else:
                    sG2[row, col] = 0.0

        # Load tiles from B into shared memory
        for idx in range(strideB):
            index = thread_id * strideB + idx
            if index < BK * BN:
                row = index // BN
                col = index % BN
                b_row = bkIdx + row
                b_col = b_start_col + col
                X = b_col // nocc
                b = b_col % nocc
                if b_row < K and b_col < N:
                    scholbar[row, col] = rcholbar[iq, ik_pr, b_row, X, b]
                else:
                    scholbar[row, col] = 0.0

        # Synchronize threads after loading
        cuda.syncthreads()

        # Compute per-thread results
        for dotIdx in range(BK):
            # Load elements from shared memory into registers
            for i in range(TM):
                regM[i] = sG2[threadRow * TM + i, dotIdx]
            for j in range(TN):
                regN[j] = scholbar[dotIdx, threadCol * TN + j]

            # Compute the dot product
            for resIdxM in range(TM):
                for resIdxN in range(TN):
                    threadResults[resIdxM, resIdxN] += regM[resIdxM] * regN[resIdxN]

        # Synchronize threads before the next iteration
        cuda.syncthreads()

    # Write the results back to the global memory matrix C
    for resIdxM in range(TM):
        c_row = a_start_row + threadRow * TM + resIdxM
        for resIdxN in range(TN):
            c_col = b_start_col + threadCol * TN + resIdxN
            X = c_col // nocc
            b = c_col % nocc
            w = c_row // nocc
            a = c_row % nocc
            if c_row < M and c_col < N:
                T2[batch_id, X, b, w, a] += threadResults[resIdxM, resIdxN]

@cuda.jit("void(complex128[:,:,:,:,:], complex128[:,:,:,:,:], complex128[:])")
def kernel_exchange_reduction(T1, T2, exx_w):
    # T1[batch_id, a, X, w, b], T2[batch_id, X, b, w, a] -> exx_w[w]
    nbatch = T1.shape[0]
    naux = T1.shape[2]
    nocc = T1.shape[1]
    nwalker = T1.shape[3]
    nocc_sq = nocc * nocc
    thread_ix = cuda.threadIdx.x
    block_ix = cuda.blockIdx.x
    if naux < nbatch:
        if block_ix > nwalker * nocc * nocc * nbatch:
            return
        walker = block_ix // (nocc_sq * nbatch)
        batch_id = (block_ix % (nocc_sq * nbatch)) // (nocc_sq)
        a = (block_ix % (nocc_sq * nbatch)) % (nocc_sq) // nocc
        b = (block_ix % (nocc_sq * nbatch)) % (nocc_sq) % nocc
        shared_array = cuda.shared.array(shape=(_block_size,), dtype=numba.complex128)
        block_size = cuda.blockDim.x
        shared_array[thread_ix] = 0.0
        for x in range(thread_ix, naux, block_size):
            shared_array[thread_ix] += T1[batch_id, a, x, walker, b] * T2[batch_id, x, b, walker, a]
        # pylint: disable=no-value-for-parameter
        cuda.syncthreads()
        nreduce = block_size // 2
        indx = nreduce
        for _ in range(0, nreduce):
            if indx == 0:
                break
            if thread_ix < indx:
                shared_array[thread_ix] += shared_array[thread_ix + indx]
            # pylint: disable=no-value-for-parameter
            cuda.syncthreads()
            indx = indx // 2
        if thread_ix == 0:
            cuda.atomic.add(exx_w.real, walker, shared_array[0].real)
            cuda.atomic.add(exx_w.imag, walker, shared_array[0].imag)
    else:
        if block_ix > nwalker * nocc * nocc * naux:
            return
        walker = block_ix // (nocc_sq * naux)
        x = (block_ix % (nocc_sq * naux)) // nocc_sq
        a = (block_ix % (nocc_sq * naux)) % nocc_sq // nocc
        b = (block_ix % (nocc_sq * naux)) % nocc_sq % nocc
        shared_array = cuda.shared.array(shape=(_block_size,), dtype=numba.complex128)
        block_size = cuda.blockDim.x
        shared_array[thread_ix] = 0.0
        for batch_id in range(thread_ix, nbatch, block_size):
            shared_array[thread_ix] += T1[batch_id, a, x, walker, b] * T2[batch_id, x, b, walker, a]
        # pylint: disable=no-value-for-parameter
        cuda.syncthreads()
        nreduce = block_size // 2
        indx = nreduce
        for _ in range(0, nreduce):
            if indx == 0:
                break
            if thread_ix < indx:
                shared_array[thread_ix] += shared_array[thread_ix + indx]
            # pylint: disable=no-value-for-parameter
            cuda.syncthreads()
            indx = indx // 2
        if thread_ix == 0:
            cuda.atomic.add(exx_w.real, walker, shared_array[0].real)
            cuda.atomic.add(exx_w.imag, walker, shared_array[0].imag)



def exx_kpt_kernel(rchol, rcholbar, Ghalf, kcubelist, kpq_mat):
    """Calculate the exchange energy for each walker.
    """
    nwalkers = Ghalf.shape[2]
    nocc = rchol.shape[2]
    naux = rchol.shape[3]
    nbasis = rchol.shape[-1]

    T1 = cp.zeros((len(kcubelist), nocc, naux, nwalkers, nocc), dtype=cp.complex128)
    T2 = cp.zeros((len(kcubelist), naux, nocc, nwalkers, nocc), dtype=cp.complex128)

    M = max(nocc * naux, nbasis)
    N = max(nocc * nwalkers, nbasis)
    blockdim_x = BN // TN
    blockdim_y = BM // TM
    threadsperblock = (blockdim_x, blockdim_y, 1)
    blockspergrid_x = (N + BN - 1) // BN
    blockspergrid_y = (M + BM - 1) // BM
    blockspergrid_z = len(kcubelist)

    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    
    get_T1[blockspergrid, threadsperblock](rchol, Ghalf, kcubelist, kpq_mat, T1)

    blockspergrid_x2 = (M + BN - 1) // BN
    blockspergrid_y2 = (N + BM - 1) // BM
    blockspergrid2 = (blockspergrid_x2, blockspergrid_y2, blockspergrid_z)
    get_T2[blockspergrid2, threadsperblock](rcholbar, Ghalf, kcubelist, kpq_mat, T2)
    cp.cuda.stream.get_current_stream().synchronize()

    exx_w = cp.zeros(nwalkers, dtype=cp.complex128)

    if naux > len(kcubelist):
        kernel_exchange_reduction[nwalkers * nocc * nocc * naux, _block_size](T1, T2, exx_w)
    else:
        kernel_exchange_reduction[nwalkers * nocc * nocc * len(kcubelist), _block_size](T1, T2, exx_w)

    cp.cuda.stream.get_current_stream().synchronize()
    return exx_w


# @cuda.jit("void(complex128[:,:,:,:], complex128[:,:,:,:], complex128[:])")
# def kernel_exchange_reduction(T1, T2, exx_w):
#     naux = T1.shape[1]
#     nocc = T1.shape[0]
#     nwalker = T1.shape[3]
#     nocc_sq = nocc * nocc
#     thread_ix = cuda.threadIdx.x
#     block_ix = cuda.blockIdx.x
#     if block_ix > nwalker * nocc * nocc:
#         return
#     walker = block_ix // nocc_sq
#     a = (block_ix % nocc_sq) // nocc
#     b = (block_ix % nocc_sq) % nocc
#     shared_array = cuda.shared.array(shape=(_block_size,), dtype=numba.complex128)
#     block_size = cuda.blockDim.x
#     shared_array[thread_ix] = 0.0
#     for x in range(thread_ix, naux, block_size):
#         shared_array[thread_ix] += T1[a, x, b, walker] * T2[walker, a, x, b]
#     # pylint: disable=no-value-for-parameter
#     cuda.syncthreads()
#     nreduce = block_size // 2
#     indx = nreduce
#     for _ in range(0, nreduce):
#         if indx == 0:
#             break
#         if thread_ix < indx:
#             shared_array[thread_ix] += shared_array[thread_ix + indx]
#         # pylint: disable=no-value-for-parameter
#         cuda.syncthreads()
#         indx = indx // 2
#     if thread_ix == 0:
#         cuda.atomic.add(exx_w.real, walker, shared_array[0].real)
#         cuda.atomic.add(exx_w.imag, walker, shared_array[0].imag)

        
# def exchange_reduction_kpt(T1, T2, exx_walker):
#     """Reduce intermediate with itself.

#     equivalent to einsum('xijw,wixj->w', T1, T2)

#     Parameters
#     ---------
#     Txiwj : np.ndarray
#         Intemediate tensor of dimension (naux, nocca/b, nwalker, nocca/b).
#     exx_walker : np.ndarray
#         Exchange contribution for all walkers in batch.
#     """
#     nwalkers = T1.shape[-1]
#     nocc = T1.shape[0]
#     blocks_per_grid = nwalkers * nocc * nocc
#     # todo add constants to config
#     # do blocks_per_grid dot products + reductions
#     # look into optimizations.
#     kernel_exchange_reduction[blocks_per_grid, _block_size](T1, T2, exx_walker)
#     cp.cuda.stream.get_current_stream().synchronize()

    
    

    
