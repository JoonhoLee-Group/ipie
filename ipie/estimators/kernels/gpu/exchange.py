
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
# Author: Fionn Malone <fmalone@google.com>
#

try:
    import cupy as cp
    import numba
    from numba import cuda
except ModuleNotFoundError:
    pass

_block_size = 512  #


@cuda.jit("void(complex128[:,:,:,:], complex128[:])")
def kernel_exchange_reduction_old(T, exx_w):
    nwalker = T.shape[0]
    naux = T.shape[1]
    nocc = T.shape[2]
    nocc_sq = nocc * nocc
    thread_ix = cuda.threadIdx.x
    block_ix = cuda.blockIdx.x
    if block_ix > nwalker * nocc * nocc:
        return
    walker = block_ix // nocc_sq
    a = (block_ix % nocc_sq) // nocc
    b = (block_ix % nocc_sq) % nocc
    shared_array = cuda.shared.array(shape=(_block_size,), dtype=numba.complex128)
    block_size = cuda.blockDim.x
    shared_array[thread_ix] = 0.0
    for x in range(thread_ix, naux, block_size):
        shared_array[thread_ix] += T[walker, x, a, b] * T[walker, x, b, a]
    cuda.syncthreads()
    nreduce = block_size // 2
    indx = nreduce
    for it in range(0, nreduce):
        if indx == 0:
            break
        if thread_ix < indx:
            shared_array[thread_ix] += shared_array[thread_ix + indx]
        cuda.syncthreads()
        indx = indx // 2
    if thread_ix == 0:
        cuda.atomic.add(exx_w.real, walker, shared_array[0].real)
        cuda.atomic.add(exx_w.imag, walker, shared_array[0].imag)


@cuda.jit("void(complex128[:,:,:,:], complex128[:])")
def kernel_exchange_reduction(T, exx_w):
    naux = T.shape[0]
    nocc = T.shape[1]
    nwalker = T.shape[2]
    nocc_sq = nocc * nocc
    thread_ix = cuda.threadIdx.x
    block_ix = cuda.blockIdx.x
    if block_ix > nwalker * nocc * nocc:
        return
    walker = block_ix // nocc_sq
    a = (block_ix % nocc_sq) // nocc
    b = (block_ix % nocc_sq) % nocc
    shared_array = cuda.shared.array(shape=(_block_size,), dtype=numba.complex128)
    block_size = cuda.blockDim.x
    shared_array[thread_ix] = 0.0
    for x in range(thread_ix, naux, block_size):
        shared_array[thread_ix] += T[x, a, walker, b] * T[x, b, walker, a]
    cuda.syncthreads()
    nreduce = block_size // 2
    indx = nreduce
    for it in range(0, nreduce):
        if indx == 0:
            break
        if thread_ix < indx:
            shared_array[thread_ix] += shared_array[thread_ix + indx]
        cuda.syncthreads()
        indx = indx // 2
    if thread_ix == 0:
        cuda.atomic.add(exx_w.real, walker, shared_array[0].real)
        cuda.atomic.add(exx_w.imag, walker, shared_array[0].imag)


def exchange_reduction_old(Twxij, exx_walker):
    """Reduce intermediate with itself.

    equivalent to einsum('wxij,wxji->w', Twxij, Twxij)

    Parameters
    ---------
    Txiwj : np.ndarray
        Intemediate tensor of dimension (naux, nocca/b, nwalker, nocca/b).
    exx_walker : np.ndarray
        Exchange contribution for all walkers in batch.
    """
    nwalkers = Twxij.shape[0]
    nocc = Twxij.shape[2]
    blocks_per_grid = nwalkers * nocc * nocc
    # todo add constants to config
    # do blocks_per_grid dot products + reductions
    # look into optimizations.
    kernel_exchange_reduction_old[blocks_per_grid, _block_size](Twxij, exx_walker)
    cp.cuda.stream.get_current_stream().synchronize()


def exchange_reduction(Txiwj, exx_walker):
    """Reduce intermediate with itself.

    equivalent to einsum('xiwj,xjwi->w', Txiwj, Txiwj)

    Parameters
    ---------
    Txiwj : np.ndarray
        Intemediate tensor of dimension (naux, nocca/b, nwalker, nocca/b).
    exx_walker : np.ndarray
        Exchange contribution for all walkers in batch.
    """
    nwalkers = Txiwj.shape[2]
    nocc = Txiwj.shape[1]
    blocks_per_grid = nwalkers * nocc * nocc
    # todo add constants to config
    # do blocks_per_grid dot products + reductions
    # look into optimizations.
    kernel_exchange_reduction[blocks_per_grid, _block_size](Txiwj, exx_walker)
    cp.cuda.stream.get_current_stream().synchronize()
