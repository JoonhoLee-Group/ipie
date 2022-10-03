import pytest

from ipie.utils.backend import arraylib as xp

from ipie.estimators.kernels import wicks as wk

@pytest.mark.gpu
def test_get_dets_single_excitation_batched():
    nwalker = 10
    ndets = 10
    nmo = 10
    G0 = xp.random.random((nwalker, nmo, nmo)).astype(xp.complex128)
    G0 += 1j * xp.random.random((nwalker, nmo, nmo))
    cre_a = [3]
    anh_a = [0]
    p = cre_a[0]
    q = anh_a[0]
    ref = xp.zeros((nwalker, ndets), dtype=xp.complex128)
    for iw in range(nwalker):
        for idet in range(ndets):
            ref[iw, idet] = G0[iw, p, q]

    from ipie.utils.misc import dotdict
    # [idet, cre] = p
    # [iexcit, cre] = [p1, p2, p3]...
    cre_ex_a = [[0], xp.array([[p]] * ndets, dtype=int)],
    anh_ex_a =  [[0], xp.array([[q]] * ndets, dtype=int)],
    occ_map_a = xp.arange(10, dtype=xp.int32),

    dets = xp.zeros_like(ref)
    wk.get_dets_singles(cre_ex_a[1], cre_ex_b[1], occ_map_a, 0, G0, dets)
    assert xp.allclose(ref, dets)
