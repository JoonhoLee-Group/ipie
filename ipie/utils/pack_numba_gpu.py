from numba import cuda


@cuda.jit("void(int32[:],int32[:],complex128[:,:],complex128[:,:,:])")
def unpack_VHS_batch_gpu(idx_i, idx_j, VHS_packed, VHS):
    nwalkers = VHS.shape[0]
    nbsf = VHS.shape[1]
    nut = round(nbsf * (nbsf + 1) / 2)
    pos = cuda.grid(1)
    pos1 = pos // nut
    pos2 = pos - pos1 * nut
    if pos1 < nwalkers and pos2 < nut:
        VHS[pos1, idx_i[pos2], idx_j[pos2]] = VHS_packed[pos1, pos2]
        VHS[pos1, idx_j[pos2], idx_i[pos2]] = VHS_packed[pos1, pos2]
