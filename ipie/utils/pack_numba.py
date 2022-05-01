import numpy
from numba import jit

def cuda_jit(rectype):
    try:
        import cupy
        assert(cupy.is_available())
        iscupyavail=True
        from numba import cuda
    except:
        iscupyavail=False
    def decorator(func):
        if not iscupyavail:
            # Return the function unchanged, not decorated.
            return func
        return cuda.jit(func, rectype)
    return decorator

@jit(nopython=True, fastmath=True)
def unpack_VHS_batch(idx_i,idx_j,VHS_packed,VHS):
    nwalkers = VHS.shape[0]
    nbsf = VHS.shape[1]
    nut = round(nbsf *(nbsf+1)/2)

    for iw in range(nwalkers):
        for i in range(nut):
            VHS[iw, idx_i[i],idx_j[i]] = VHS_packed[iw,i]
            VHS[iw, idx_j[i],idx_i[i]] = VHS_packed[iw,i]

    return
from numba import cuda
@cuda.jit('void(int32[:],int32[:],complex128[:,:],complex128[:,:,:])')
def unpack_VHS_batch_gpu(idx_i,idx_j,VHS_packed,VHS):
    nwalkers = VHS.shape[0]
    nbsf = VHS.shape[1]
    nut = round(nbsf *(nbsf+1)/2)
    pos = cuda.grid(1)
    pos1 = pos // nut
    pos2 = pos - pos1 * nut
    if (pos1 < nwalkers and pos2 < nut):
        VHS[pos1, idx_i[pos2],idx_j[pos2]] = VHS_packed[pos1,pos2]
        VHS[pos1, idx_j[pos2],idx_i[pos2]] = VHS_packed[pos1,pos2]
