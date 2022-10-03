try:
    import numba
    from numba import cuda
except ModuleNotFoundError:
    pass

_block_size = 512  #

@cuda.jit("void(int[:,:],  int[:,:], int[:], int, complex128[:,:,:,:], complex128[:])")
def kernel_get_dets_singles(cre, anh, mapping, offset, G0, dets):
    """Get overlap from singly excited Slater-Determinants.

    Parameters
    ----------
    cre : cp.ndarray
        Array containing orbitals excitations of occupied.
    anh : cp.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : cp.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : cp.ndarray
        (Half rotated) batched Green's function.
    dets : cp.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """
    ndets = anh.shape[0]
    nwalkers = G0.shape[0]
    pos = cuda.grid(1)
    if pos > nwalker * ndets:
        return
    idet = pos // nwalkers
    iwalker = pos % nwalkers
    p = mapping[cre[idet, 0]] + offset
    q = anh[idet, 0] + offset
    dets[iwalker, idet] = G0[iwalker, p, q]

def get_dets_singles(cre, anh, mapping, offset, G0, dets):
    ndets = anh.shape[0]
    nwalkers = G0.shape[0]
    blocks_per_grid = math.ceil(ndets*nwalkers/_block_size)
    kernel_get_dets_singles[block_per_grid, _block_size](cre, anh, mapping, offset, G0, dets)
