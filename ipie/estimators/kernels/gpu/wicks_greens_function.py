import math

try:
    from numba import cuda
except ModuleNotFoundError:
    pass

from ipie.utils.backend import synchronize

_block_size = 512  #


# @cuda.jit("void(int32[:,:],  int32[:,:], int32[:], complex128[:,:], complex128[:,:,:])")
# def kernel_reduce_CI_singles(
    # cre,
    # anh,
    # mapping,
    # phases,
    # CI,
# ) -> None:
    # """Reduction to CI intermediate for singles.

    # Parameters
    # ----------
    # cre : np.ndarray
        # Array containing orbitals excitations of occupied.
    # anh : np.ndarray
        # Array containing orbitals excitations to virtuals.
    # mapping : np.ndarray
        # Map original (occupied) orbital to index in compressed form.
    # phases : np.ndarray
        # Phase factors.
    # CI : np.ndarray
        # Output array for CI intermediate.

    # Returns
    # -------
    # None
    # """
    # # ndets = anh.shape[0]
    # # nwalkers = G0.shape[0]
    # # pos = cuda.grid(1)
    # # idet = pos // nwalkers
    # # iwalker = pos % nwalkers
    # # if idet < ndets and iwalker < nwalkers:
    # # p = mapping[cre[idet, 0]] + offset
    # # q = anh[idet, 0] + offset
    # # dets[iwalker, idet] = G0[iwalker, p, q]
    # ndets = cre.shape[0]
    # nwalkers = phases.shape[0]
    # for iw in range(nwalkers):
        # for idet in range(ndets):
            # p = mapping[ps[idet]]
            # q = qs[idet]
            # CI[iw, q, p] += phases[iw, idet]


def reduce_CI_singles(cre, anh, mapping, phases, CI):
    # ndets = anh.shape[0]
    # nwalkers = CI.shape[0]
    # blocks_per_grid = math.ceil(ndets * nwalkers / _block_size)
    assert False
    # kernel_reduce_CI_singles[blocks_per_grid, _block_size](
        # cre, anh, mapping, offset, G0, dets
    # )
    # synchronize()
