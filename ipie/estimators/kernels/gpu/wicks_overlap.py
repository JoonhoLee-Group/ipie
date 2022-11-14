import math

try:
    import numba
    from numba import cuda
except ModuleNotFoundError:
    pass

from ipie.utils.backend import synchronize
from ipie.utils.backend import arraylib as xp

_block_size = 512  #


@cuda.jit(
    "void(int32[:,:],  int32[:,:], int32[:], int32, complex128[:,:,:], complex128[:,:])"
)
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
    idet = pos // nwalkers
    iwalker = pos % nwalkers
    if idet < ndets and iwalker < nwalkers:
        p = mapping[cre[idet, 0]] + offset
        q = anh[idet, 0] + offset
        dets[iwalker, idet] = G0[iwalker, p, q]


@cuda.jit(
    "void(int32[:,:],  int32[:,:], int32[:], int32, complex128[:,:,:], complex128[:,:])"
)
def kernel_get_dets_doubles(cre, anh, mapping, offset, G0, dets):
    """Get overlap from double excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    dets : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """
    ndets = anh.shape[0]
    nwalkers = G0.shape[0]
    pos = cuda.grid(1)
    idet = pos // nwalkers
    iwalker = pos % nwalkers
    if idet < ndets and iwalker < nwalkers:
        p = mapping[cre[idet, 0]] + offset
        r = mapping[cre[idet, 1]] + offset
        q = anh[idet, 0] + offset
        s = anh[idet, 1] + offset
        dets[iwalker, idet] = (
            G0[iwalker, p, q] * G0[iwalker, r, s]
            - G0[iwalker, p, s] * G0[iwalker, r, q]
        )


@cuda.jit(
    "void(int32[:,:],  int32[:,:], int32[:], int32, complex128[:,:,:], complex128[:,:])"
)
def kernel_get_dets_triples(
    cre,
    anh,
    mapping,
    offset,
    G0,
    dets,
):
    """Get overlap from triply excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    dets : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """
    ndets = anh.shape[0]
    nwalkers = G0.shape[0]
    pos = cuda.grid(1)
    idet = pos // nwalkers
    iwalker = pos % nwalkers
    if idet < ndets and iwalker < nwalkers:
        ps, qs = mapping[cre[idet, 0]] + offset, anh[idet, 0] + offset
        rs, ss = mapping[cre[idet, 1]] + offset, anh[idet, 1] + offset
        ts, us = mapping[cre[idet, 2]] + offset, anh[idet, 2] + offset
        dets[iwalker, idet] = (
            G0[iwalker, ps, qs]
            * (
                G0[iwalker, rs, ss] * G0[iwalker, ts, us]
                - G0[iwalker, rs, us] * G0[iwalker, ts, ss]
            )
            - G0[iwalker, ps, ss]
            * (
                G0[iwalker, rs, qs] * G0[iwalker, ts, us]
                - G0[iwalker, rs, us] * G0[iwalker, ts, qs]
            )
            + G0[iwalker, ps, us]
            * (
                G0[iwalker, rs, qs] * G0[iwalker, ts, ss]
                - G0[iwalker, rs, ss] * G0[iwalker, ts, qs]
            )
        )


@cuda.jit(
    "void(int32[:,:],  int32[:,:], int32[:], int32, complex128[:,:,:], complex128[:,:,:,:])"
)
def kernel_get_dets_nfold(cre, anh, mapping, offset, G0, det_mat):
    """Get overlap from n-fold excited Slater-Determinants.

    Parameters
    ----------
    cre : np.ndarray
        Array containing orbitals excitations of occupied.
    anh : np.ndarray
        Array containing orbitals excitations to virtuals.
    mapping : np.ndarray
        Map original (occupied) orbital to index in compressed form.
    offset : int
        Offset for frozen core.
    G0 : np.ndarray
        (Half rotated) batched Green's function.
    det_mat : np.ndarray
        Output array of determinants <D_I|phi>.

    Returns
    -------
    None
    """
    ndets = len(cre)
    nwalkers = G0.shape[0]
    nex = cre.shape[-1]
    pos = cuda.grid(1)
    # pos = nwalkers * ndets * nex * nex
    iwalker = pos % nwalkers
    idet = ((pos - iwalker) // nwalkers) % ndets
    iex = ((pos - idet * nwalkers - iwalker) // (nwalkers * ndets)) % nex
    jex = (
        (pos - iex * nex * nwalkers - idet * nwalkers - iwalker)
        // (nwalkers * ndets * nex)
    ) % nex
    if idet < ndets and iwalker < nwalkers and iex < nex:
        p = mapping[cre[idet, iex]] + offset
        q = anh[idet, iex] + offset
        det_mat[iwalker, idet, iex, iex] = G0[iwalker, p, q]
        if jex < nex and jex > iex:
            r = mapping[cre[idet, jex]] + offset
            s = anh[idet, jex] + offset
            det_mat[iwalker, idet, iex, jex] = G0[iwalker, p, s]
            det_mat[iwalker, idet, jex, iex] = G0[iwalker, r, q]


def get_dets_singles(cre, anh, mapping, offset, G0, dets):
    ndets = anh.shape[0]
    nwalkers = G0.shape[0]
    blocks_per_grid = math.ceil(ndets * nwalkers / _block_size)
    kernel_get_dets_singles[blocks_per_grid, _block_size](
        cre, anh, mapping, offset, G0, dets
    )
    synchronize()


def get_dets_doubles(cre, anh, mapping, offset, G0, dets):
    ndets = anh.shape[0]
    nwalkers = G0.shape[0]
    blocks_per_grid = math.ceil(ndets * nwalkers / _block_size)
    kernel_get_dets_doubles[blocks_per_grid, _block_size](
        cre, anh, mapping, offset, G0, dets
    )
    synchronize()


def get_dets_triples(cre, anh, mapping, offset, G0, dets):
    ndets = anh.shape[0]
    nwalkers = G0.shape[0]
    blocks_per_grid = math.ceil(ndets * nwalkers / _block_size)
    kernel_get_dets_triples[blocks_per_grid, _block_size](
        cre, anh, mapping, offset, G0, dets
    )
    synchronize()


def get_dets_nfold(cre, anh, mapping, offset, G0, det_mat_buffer, dets):
    ndets = anh.shape[0]
    nwalkers = G0.shape[0]
    nex = cre.shape[-1]
    blocks_per_grid = math.ceil(ndets * nwalkers * nex * nex / _block_size)
    kernel_get_dets_nfold[blocks_per_grid, _block_size](
        cre, anh, mapping, offset, G0, det_mat_buffer
    )
    synchronize()
    dets[:] = xp.linalg.det(det_mat_buffer)
    synchronize()
