import numpy
from numba import jit


@jit(nopython=True, fastmath=True)
def unpack_VHS_batch(idx_i, idx_j, VHS_packed, VHS):
    nwalkers = VHS.shape[0]
    nbsf = VHS.shape[1]
    nut = round(nbsf * (nbsf + 1) / 2)

    for iw in range(nwalkers):
        for i in range(nut):
            VHS[iw, idx_i[i], idx_j[i]] = VHS_packed[iw, i]
            VHS[iw, idx_j[i], idx_i[i]] = VHS_packed[iw, i]

    return
