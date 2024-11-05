from numba import cuda, complex128
import cupy as cp

TPB = 16

BM = 32
BN = 64
BK = 8
TM = 4
TN = 4

@cuda.jit("void(complex128[:, :, :, :, :], complex128[:, :, :], complex128[:, :, :], int64[:, :], complex128[:, :, :, :, :], complex128[:, :, :, :, :])")
def kernel_VHS_construction(chol, xshifted, xshifted_conj, ikpq_mat, VHS1, VHS2):
    """
    Construct the VHS matrix in the symmetrized form using shared memory.
    """
    # q * k: batched index I
    # J: nwalkers
    # L: naux
    # K: nbasis**2
    nk = chol.shape[1]
    naux = chol.shape[0]
    nbasis = chol.shape[-1]
    nbasis_sq = nbasis**2
    nwalker = xshifted.shape[0]
    J = nwalker
    L = naux
    K = nbasis_sq
    iqk = cuda.blockIdx.z  # I
    row = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y # J
    col = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x # K
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    p = col // nbasis
    r = col % nbasis

    iq = iqk // nk
    ik = iqk % nk
    # iq_real = qset[iq]
    ikpq = ikpq_mat[iq, ik]

    # Allocate shared memory for tiles of chol and xshifted
    sx = cuda.shared.array(shape=(TPB, TPB), dtype=complex128)
    # sxbar = cuda.shared.array(shape=(TPB, TPB), dtype=complex128)
    schol = cuda.shared.array(shape=(TPB, TPB), dtype=complex128)
    # Initialize the accumulator
    tmp1 = complex128(0.0)
    tmp2 = complex128(0.0)
    for l in range(0, L, TPB):
        if row < J and (l + tx) < L:
            sx[ty, tx] = xshifted[row, l + tx, iq]
        else:
            sx[ty, tx] = complex128(0.0)
        if (l + ty) < L and col < K:
            schol[ty, tx] = chol[l + ty, ik, p, iq, r]
        else:
            schol[ty, tx] = complex128(0.0)
        cuda.syncthreads()
        for k in range(TPB):
            tmp1 += sx[ty, k] * schol[k, tx]
        cuda.syncthreads()

    if row < J and col < K:
        VHS1[row, ik, p, ikpq, r] += tmp1

    for l in range(0, L, TPB):
        if row < J and (l + tx) < L:
            sx[ty, tx] = xshifted_conj[row, l + tx, iq]
        else:
            sx[ty, tx] = complex128(0.0)
        cuda.syncthreads()
        for k in range(TPB):
            tmp2 += sx[ty, k] * schol[k, tx].conjugate()
        cuda.syncthreads()

    if row < J and col < K:
        VHS2[row, ikpq, r, ik, p] += tmp2

def call_kernel_VHS_construction(chol, xshifted, xshifted_conj, naux, nk, nbasis, nwalker, ikpq_mat, VHS1, VHS2):
    threadsperblock = (TPB, TPB, 1)
    grid_x_max = max(naux, nbasis**2)
    grid_y_max = max(nwalker, naux)
    blockspergrid_x = (grid_x_max + TPB - 1) // TPB
    blockspergrid_y = (grid_y_max + TPB - 1) // TPB
    blockspergrid_z = ikpq_mat.shape[0] * nk
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    kernel_VHS_construction[blockspergrid, threadsperblock](chol, xshifted, xshifted_conj, ikpq_mat, VHS1, VHS2)
    cp.cuda.stream.get_current_stream().synchronize()

@cuda.jit("void(complex128[:, :, :, :, :], complex128[:, :, :], int64[:, :], complex128[:, :, :, :, :])")
def kernel_VHS_construction1(chol, xshifted, ikpq_mat, VHS):
    """
    Construct the VHS matrix in the symmetrized form using shared memory.
    """
    # q * k: batched index I
    # M: nwalkers
    # K: naux
    # N: nbasis**2
    nk = chol.shape[1]
    naux = chol.shape[0]
    nbasis = chol.shape[-1]
    nbasis_sq = nbasis**2
    nwalker = xshifted.shape[0]
    M = nwalker
    K = naux
    N = nbasis_sq
    iqk = cuda.blockIdx.z  # I
    cRow = cuda.blockIdx.y
    cCol = cuda.blockIdx.x
    threadCol = cuda.threadIdx.x
    threadRow = cuda.threadIdx.y

    threadsPerRow = BN // TN
    thread_id = threadRow * threadsPerRow + threadCol

    # p = col // nbasis
    # r = col % nbasis

    iq = iqk // nk
    ik = iqk % nk
    # iq_real = qset[iq]
    ikpq = ikpq_mat[iq, ik]

    # Allocate shared memory for tiles of chol and xshifted
    sx = cuda.shared.array(shape=(BM, BK), dtype=complex128)
    schol = cuda.shared.array(shape=(BK, BN), dtype=complex128)
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
                if a_row < M and a_col < K:
                    sx[row, col] = xshifted[a_row, a_col, iq]
                else:
                    sx[row, col] = 0.0

        # Load tiles from B into shared memory
        for idx in range(strideB):
            index = thread_id * strideB + idx
            if index < BK * BN:
                row = index // BN
                col = index % BN
                b_row = bkIdx + row
                b_col = b_start_col + col
                p = b_col // nbasis
                r = b_col % nbasis
                if b_row < K and b_col < N:
                    schol[row, col] = chol[b_row, ik, p, iq, r]
                else:
                    schol[row, col] = 0.0

        # Synchronize threads after loading
        cuda.syncthreads()

        # Compute per-thread results
        for dotIdx in range(BK):
            # Load elements from shared memory into registers
            for i in range(TM):
                regM[i] = sx[threadRow * TM + i, dotIdx]
            for j in range(TN):
                regN[j] = schol[dotIdx, threadCol * TN + j]

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
            p = c_col // nbasis
            r = c_col % nbasis
            if c_row < M and c_col < N:
                VHS[c_row, ik, p, ikpq, r] += threadResults[resIdxM, resIdxN]

@cuda.jit("void(complex128[:, :, :, :, :], complex128[:, :, :], int64[:, :], complex128[:, :, :, :, :])")
def kernel_VHS_construction2(chol, xshifted, ikpq_mat, VHS):
    """
    Construct the VHS matrix in the symmetrized form using shared memory.
    """
    # q * k: batched index I
    # M: nwalkers
    # K: naux
    # N: nbasis**2
    nk = chol.shape[1]
    naux = chol.shape[0]
    nbasis = chol.shape[-1]
    nbasis_sq = nbasis**2
    nwalker = xshifted.shape[0]
    M = nwalker
    K = naux
    N = nbasis_sq
    iqk = cuda.blockIdx.z  # I
    cRow = cuda.blockIdx.y
    cCol = cuda.blockIdx.x
    threadCol = cuda.threadIdx.x
    threadRow = cuda.threadIdx.y

    threadsPerRow = BN // TN
    thread_id = threadRow * threadsPerRow + threadCol

    # p = col // nbasis
    # r = col % nbasis

    iq = iqk // nk
    ik = iqk % nk
    # iq_real = qset[iq]
    ikpq = ikpq_mat[iq, ik]

    # Allocate shared memory for tiles of chol and xshifted
    sx = cuda.shared.array(shape=(BM, BK), dtype=complex128)
    schol = cuda.shared.array(shape=(BK, BN), dtype=complex128)
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
                if a_row < M and a_col < K:
                    sx[row, col] = xshifted[a_row, a_col, iq]
                else:
                    sx[row, col] = 0.0

        # Load tiles from B into shared memory
        for idx in range(strideB):
            index = thread_id * strideB + idx
            if index < BK * BN:
                row = index // BN
                col = index % BN
                b_row = bkIdx + row
                b_col = b_start_col + col
                p = b_col // nbasis
                r = b_col % nbasis
                if b_row < K and b_col < N:
                    schol[row, col] = chol[b_row, ik, p, iq, r]
                else:
                    schol[row, col] = 0.0

        # Synchronize threads after loading
        cuda.syncthreads()

        # Compute per-thread results
        for dotIdx in range(BK):
            # Load elements from shared memory into registers
            for i in range(TM):
                regM[i] = sx[threadRow * TM + i, dotIdx]
            for j in range(TN):
                regN[j] = schol[dotIdx, threadCol * TN + j].conjugate()

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
            p = c_col // nbasis
            r = c_col % nbasis
            if c_row < M and c_col < N:
                VHS[c_row, ikpq, r, ik, p] += threadResults[resIdxM, resIdxN]

def call_kernel_VHS_construction1(chol, xshifted, naux, nk, nbasis, nwalker, ikpq_mat, VHS):
    # grid_x_max = max(naux, nbasis**2)
    # grid_y_max = max(nwalker, naux)
    M = max(nwalker, naux)
    N = max(naux, nbasis**2)
    block_dim_x = BN // TN
    block_dim_y = BM // TM
    blockspergrid_x = (N + BN - 1) // BN
    blockspergrid_y = (M + BM - 1) // BM
    blockspergrid_z = ikpq_mat.shape[0] * nk
    # print(blockspergrid_x, blockspergrid_y, blockspergrid_z)
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    threadsperblock = (block_dim_x, block_dim_y, 1)
    kernel_VHS_construction1[blockspergrid, threadsperblock](chol, xshifted, ikpq_mat, VHS)
    cp.cuda.stream.get_current_stream().synchronize()

def call_kernel_VHS_construction2(chol, xshifted, naux, nk, nbasis, nwalker, ikpq_mat, VHS):
    # grid_x_max = max(naux, nbasis**2)
    # grid_y_max = max(nwalker, naux)
    M = max(nwalker, naux)
    N = max(naux, nbasis**2)
    block_dim_x = BN // TN
    block_dim_y = BM // TM
    blockspergrid_x = (N + BN - 1) // BN
    blockspergrid_y = (M + BM - 1) // BM
    blockspergrid_z = ikpq_mat.shape[0] * nk
    # print(blockspergrid_x, blockspergrid_y, blockspergrid_z)
    blockspergrid = (blockspergrid_x, blockspergrid_y, blockspergrid_z)
    threadsperblock = (block_dim_x, block_dim_y, 1)
    kernel_VHS_construction2[blockspergrid, threadsperblock](chol, xshifted, ikpq_mat, VHS)
    cp.cuda.stream.get_current_stream().synchronize()