import numpy as np
import cupy as cp
import time
import numba
from numba import cuda


BLOCK_SIZE = 512

def current_gpu(rchola, rcholb, Ghalfa, Ghalfb):
    nwalkers = Ghalfa.shape[0]
    nalpha = Ghalfa.shape[1]
    nbasis = Ghalfa.shape[2]
    nchol = rchola.shape[0]
    _Ghalfa = Ghalfa.reshape(nwalkers, nalpha*nbasis)
    _Ghalfb = Ghalfb.reshape(nwalkers, nalpha*nbasis)
    _rchola = rcholb.reshape(nchol, nalpha*nbasis)
    _rcholb = rcholb.reshape(nchol, nalpha*nbasis)
    Xa = _rchola.dot(_Ghalfa.real.T) + 1.j * _rchola.dot(_Ghalfa.imag.T)
    Xb = _rcholb.dot(_Ghalfb.real.T) + 1.j * _rcholb.dot(_Ghalfb.imag.T)
    ecoul = cp.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += cp.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2. * cp.einsum("xw,xw->w", Xa, Xb, optimize=True)

    _Ghalfa = Ghalfa.reshape(nwalkers, nalpha, nbasis)
    _Ghalfb = Ghalfb.reshape(nwalkers, nalpha, nbasis)

    _rchola = rcholb.reshape(nchol, nalpha, nbasis)
    _rcholb = rcholb.reshape(nchol, nalpha, nbasis)

    Txij = cp.einsum("xim,wjm->wxji", _rchola, _Ghalfa)
    exx  = cp.einsum("wxji,wxij->w",Txij,Txij)
    Txij = cp.einsum("xim,wjm->wxji", _rcholb, _Ghalfb)
    exx += cp.einsum("wxji,wxij->w",Txij,Txij)

    return ecoul, exx

# @cuda.jit('void(float64[:,:,:,:], float64[:])')
# def exchange_kernel(T, exx_w):
    # nwalker = T.shape[0]
    # naux = T.shape[1]
    # nocc = T.shape[2]
    # nocc_sq = nocc * nocc
    # thread_ix = cuda.threadIdx.x
    # block_ix = cuda.blockIdx.x
    # walker = block_ix // nocc_sq
    # a = (block_ix % nocc_sq) // nocc
    # b = (block_ix % nocc_sq) % nocc
    # shared_array = cuda.shared.array(shape=(BLOCK_SIZE,), dtype=numba.float64)
    # block_size = cuda.blockDim.x
    # shared_array[thread_ix] = 0.0
    # for x in range(thread_ix, naux, block_size):
        # shared_array[thread_ix] += T[walker, x, a, b] * T[walker, x, b, a]
    # cuda.syncthreads()
    # nreduce = block_size // 2
    # indx = nreduce
    # for it in range(0, nreduce):
        # if indx == 0:
            # break
        # if thread_ix < indx:
            # shared_array[thread_ix] += shared_array[thread_ix + indx]
        # cuda.syncthreads()
        # indx = indx // 2
    # if thread_ix == 0:
        # cuda.atomic.add(exx_w, walker, shared_array[0])

# @cuda.jit('void(complex128[:,:,:,:], complex128[:])')
# def exchange_kernel(T, exx_w):
    # nwalker = T.shape[0]
    # naux = T.shape[1]
    # nocc = T.shape[2]
    # nocc_sq = nocc * nocc
    # thread_ix = cuda.threadIdx.x
    # block_ix = cuda.blockIdx.x
    # walker = block_ix // nocc_sq
    # a = (block_ix % nocc_sq) // nocc
    # b = (block_ix % nocc_sq) % nocc
    # shared_array = cuda.shared.array(shape=(BLOCK_SIZE,), dtype=numba.complex128)
    # block_size = cuda.blockDim.x
    # shared_array[thread_ix] = 0.0
    # for x in range(thread_ix, naux, block_size):
        # shared_array[thread_ix] += T[walker, x, a, b] * T[walker, x, b, a]
    # cuda.syncthreads()
    # nreduce = block_size // 2
    # indx = nreduce
    # for it in range(0, nreduce):
        # if indx == 0:
            # break
        # if thread_ix < indx:
            # shared_array[thread_ix] += shared_array[thread_ix + indx]
        # cuda.syncthreads()
        # indx = indx // 2
    # if thread_ix == 0:
        # cuda.atomic.add(exx_w.real, walker, shared_array[0].real)
        # cuda.atomic.add(exx_w.imag, walker, shared_array[0].imag)


def new_gpu(rchola, rcholb, Ghalfa, Ghalfb):
    nwalkers = Ghalfa.shape[0]
    nalpha = Ghalfa.shape[1]
    nbasis = Ghalfa.shape[2]
    nchol = rchola.shape[0]
    _Ghalfa = Ghalfa.reshape(nwalkers, nalpha*nbasis)
    _Ghalfb = Ghalfb.reshape(nwalkers, nalpha*nbasis)
    _rchola = rcholb.reshape(nchol, nalpha*nbasis)
    _rcholb = rcholb.reshape(nchol, nalpha*nbasis)
    Xa = _rchola.dot(_Ghalfa.real.T) + 1.j * _rchola.dot(_Ghalfa.imag.T)
    Xb = _rcholb.dot(_Ghalfb.real.T) + 1.j * _rcholb.dot(_Ghalfb.imag.T)
    ecoul = cp.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += cp.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2. * cp.einsum("xw,xw->w", Xa, Xb, optimize=True)

    _Ghalfa = Ghalfa.reshape(nwalkers, nalpha, nbasis)
    _Ghalfb = Ghalfb.reshape(nwalkers, nalpha, nbasis)

    _rchola = rcholb.reshape(nchol, nalpha, nbasis)
    _rcholb = rcholb.reshape(nchol, nalpha, nbasis)

    Txij = cp.einsum("xim,wjm->wxji", _rchola, _Ghalfa)
    exx  = cp.einsum("wxji,wxij->w", Txij, Txij)
    Txij = cp.einsum("xim,wjm->wxji", _rcholb, _Ghalfb)
    exx += cp.einsum("wxji,wxij->w", Txij, Txij)

    return ecoul, exxcache

def current_cpu(rchola, rcholb, Ghalfa, Ghalfb):
    nwalkers = Ghalfa.shape[0]
    nalpha = Ghalfa.shape[1]
    nbasis = Ghalfa.shape[2]
    nchol = rchola.shape[0]
    _Ghalfa = Ghalfa.reshape(nwalkers, nalpha*nbasis)
    _Ghalfb = Ghalfb.reshape(nwalkers, nalpha*nbasis)
    _rchola = rcholb.reshape(nchol, nalpha*nbasis)
    _rcholb = rcholb.reshape(nchol, nalpha*nbasis)
    Xa = _rchola.dot(_Ghalfa.real.T) + 1.j * _rchola.dot(_Ghalfa.imag.T)
    Xb = _rcholb.dot(_Ghalfb.real.T) + 1.j * _rcholb.dot(_Ghalfb.imag.T)
    ecoul = np.einsum("xw,xw->w", Xa, Xa, optimize=True)
    ecoul += np.einsum("xw,xw->w", Xb, Xb, optimize=True)
    ecoul += 2. * np.einsum("xw,xw->w", Xa, Xb, optimize=True)

    _Ghalfa = Ghalfa.reshape(nwalkers, nalpha, nbasis)
    _Ghalfb = Ghalfb.reshape(nwalkers, nalpha, nbasis)

    _rchola = rcholb.reshape(nchol, nalpha, nbasis)
    _rcholb = rcholb.reshape(nchol, nalpha, nbasis)

    Txij = np.einsum("xim,wjm->wxji", _rchola, _Ghalfa)
    exx  = np.einsum("wxji,wxij->w",Txij,Txij)
    Txij = np.einsum("xim,wjm->wxji", _rcholb, _Ghalfb)
    exx += np.einsum("wxji,wxij->w",Txij,Txij)

    return ecoul, exx

def exchange_kernel_cpu(T, out):
    out[:] = np.einsum("wxij,wxji->w", T, T, optimize=True)

def exchange_kernel_gpu(T, out):
    out[:] = cp.einsum("wxij,wxji->w", T, T, optimize=True)
    cp.cuda.stream.get_current_stream().synchronize()

def exchange_kernel_gpu_numba(T, out):
    nwalkers = T.shape[0]
    nocc = T.shape[2]
    blocks_per_grid = nwalkers * nocc * nocc
    exchange_kernel[blocks_per_grid, BLOCK_SIZE](T, out)
    cp.cuda.stream.get_current_stream().synchronize()

nchol = 200
nbasis = 450
nocc = 100
nwalkers = 20

np.random.seed(7)
rchola = np.random.random((nchol, nocc, nbasis))
rcholb = np.random.random((nchol, nocc, nbasis))
print(rchola.nbytes/1024**3)
print(rcholb.nbytes/1024**3)
ghalfa = (
        np.random.random((nwalkers, nocc, nbasis)) +
        1j * np.random.random((nwalkers, nocc, nbasis))
        )
ghalfb = (
        np.random.random((nwalkers, nocc, nbasis)) +
        1j * np.random.random((nwalkers, nocc, nbasis))
        )
start = time.time()
rchola_cp = cp.asarray(rchola)
rcholb_cp = cp.asarray(rcholb)
ghalfa_cp = cp.asarray(ghalfa)
ghalfb_cp = cp.asarray(ghalfb)
from ipie.estimators.kernels import exchange_reduction
for n in range(7, 8):
    nocc = 40
    nchol = nocc * 5 * n
    _X = np.random.normal(size=(nwalkers*nchol*nocc*nocc))
    _X = _X.reshape(nwalkers, nchol, nocc, nocc)
    T = _X + 1j*_X
    T_cp =  cp.asarray(T)
    out = np.zeros((nwalkers), dtype=np.complex128)
    exchange_kernel_cpu(T, out)
    out_cp_cupy = cp.zeros((nwalkers), dtype=np.complex128)
    exchange_kernel_gpu(T_cp, out_cp_cupy)
    out_cp = cp.zeros((nwalkers), dtype=np.complex128)
    exchange_reduction(T_cp, out_cp)
    start = time.time()
    exchange_kernel_cpu(T, out)
    cpu_time = time.time()-start
    out_cp_cupy = cp.zeros((nwalkers), dtype=np.complex128)
    start = time.time()
    free_bytes, total_bytes = cp.cuda.Device().mem_info
    used_bytes = total_bytes - free_bytes
    print("# {:4.3f} GB out of {:4.3f} GB memory on GPU".format(used_bytes/1024**3,total_bytes/1024**3))
    exchange_kernel_gpu(T_cp, out_cp_cupy)
    print("here")
    cupy_time = time.time()-start
    out_cp = cp.zeros((nwalkers), dtype=np.complex128)
    free_bytes, total_bytes = cp.cuda.Device().mem_info
    used_bytes = total_bytes - free_bytes
    print("# {:4.3f} GB out of {:4.3f} GB memory on GPU".format(used_bytes/1024**3,total_bytes/1024**3))
    start = time.time()
    exchange_reduction(T_cp, out_cp)
    print("here")
    numba_time = time.time()-start
    print(n, nchol, nocc, cpu_time, cupy_time, numba_time,
          np.max(np.abs((out_cp.get()-out_cp_cupy.get()))))
