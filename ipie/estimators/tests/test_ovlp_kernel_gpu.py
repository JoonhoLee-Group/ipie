import pytest

import numpy as np

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize

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
    cre_ex_a = [[0], xp.array([[p]] * ndets, dtype=int)]
    anh_ex_a =  [[0], xp.array([[q]] * ndets, dtype=int)]
    occ_map_a = xp.arange(10, dtype=xp.int32)

    dets = xp.zeros_like(ref)
    wk.get_dets_singles(cre_ex_a[1], anh_ex_a[1], occ_map_a, 0, G0, dets)
    assert xp.allclose(ref, dets)


@pytest.mark.gpu
def test_get_dets_double_excitation_batched():
    nwalker = 10
    ndets = 10
    nmo = 10
    G0 = xp.random.random((nwalker, nmo, nmo)).astype(xp.complex128)
    G0 += 1j * xp.random.random((nwalker, nmo, nmo))
    cre_a = [3, 7]
    anh_a = [0, 2]
    p = cre_a[0]
    q = anh_a[0]
    r = cre_a[1]
    s = anh_a[1]
    ref = xp.zeros((nwalker, ndets), dtype=xp.complex128)
    for iw in range(nwalker):
        for idet in range(ndets):
            ref[iw, idet] = G0[iw, p, q] * G0[iw, r, s] - G0[iw, p, s] * G0[iw, r, q]

    from ipie.utils.misc import dotdict

    trial = dotdict(
        {
            "cre_ex_a": [[0], [0], xp.array([cre_a] * ndets, dtype=int)],
            "anh_ex_a": [[0], [0], xp.array([anh_a] * ndets, dtype=int)],
            "cre_ex_b": [[0], [0], xp.array([cre_a] * ndets, dtype=int)],
            "anh_ex_b": [[0], [0], xp.array([anh_a] * ndets, dtype=int)],
            "occ_map_a": xp.arange(10, dtype=xp.int32),
            "occ_map_b": xp.arange(10, dtype=xp.int32),
            "nfrozen": 0,
        }
    )

    dets = xp.zeros_like(ref)
    wk.get_dets_doubles(
            trial.cre_ex_a[2],
            trial.anh_ex_a[2],
            trial.occ_map_a,
            0,
            G0,
            dets)
    assert xp.allclose(ref, dets)

@pytest.mark.gpu
def test_get_dets_triple_excitation_batched():
    nwalker = 10
    ndets = 10
    nmo = 10
    xp.random.seed(7)
    G0 = xp.random.random((nwalker, nmo, nmo)).astype(xp.complex128)
    G0 += 1j * xp.random.random((nwalker, nmo, nmo))
    cre_a = [3, 7, 9]
    anh_a = [0, 1, 2]
    p = cre_a[0]
    q = anh_a[0]
    r = cre_a[1]
    s = anh_a[1]
    t = cre_a[2]
    u = anh_a[2]
    ref = xp.zeros((nwalker, ndets), dtype=xp.complex128)
    for iw in range(nwalker):
        for idet in range(ndets):
            G0a = G0[iw]
            ovlp_a = G0a[p, q] * (G0a[r, s] * G0a[t, u] - G0a[r, u] * G0a[t, s])
            ovlp_a -= G0a[p, s] * (G0a[r, q] * G0a[t, u] - G0a[r, u] * G0a[t, q])
            ovlp_a += G0a[p, u] * (G0a[r, q] * G0a[t, s] - G0a[r, s] * G0a[t, q])
            ref[iw, idet] = ovlp_a

    from ipie.utils.misc import dotdict

    trial = dotdict(
        {
            "cre_ex_a": [[0], [0], [0], xp.array([cre_a] * ndets, dtype=int)],
            "anh_ex_a": [[0], [0], [0], xp.array([anh_a] * ndets, dtype=int)],
            "cre_ex_b": [[0], [0], [0], xp.array([cre_a] * ndets, dtype=int)],
            "anh_ex_b": [[0], [0], [0], xp.array([anh_a] * ndets, dtype=int)],
            "occ_map_b": xp.arange(10, dtype=xp.int32),
            "occ_map_a": xp.arange(10, dtype=xp.int32),
            "nfrozen": 0,
        }
    )

    dets = xp.zeros_like(ref)
    wk.get_dets_triples(
            trial.cre_ex_a[3],
            trial.anh_ex_a[3],
            trial.occ_map_a,
            0,
            G0,
            dets)
    assert xp.allclose(ref, dets)


@pytest.mark.unit
def test_get_dets_nfold_excitation_batched():
    nwalker = 10
    ndets = 10
    nmo = 18
    G0 = xp.random.random((nwalker, nmo, nmo)).astype(xp.complex128)
    G0 += 1j * xp.random.random((nwalker, nmo, nmo))
    cre_a = [9, 10, 11, 13, 15]
    anh_a = [0, 1, 2, 4, 8]
    ref = xp.zeros((nwalker, ndets), dtype=xp.complex128)
    nex_a = 5
    for iw in range(nwalker):
        G0a = G0[iw]
        for idet in range(ndets):
            det_a = xp.zeros((nex_a, nex_a), dtype=xp.complex128)
            for iex in range(nex_a):
                p = cre_a[iex]
                q = anh_a[iex]
                det_a[iex, iex] = G0a[p, q]
                for jex in range(iex + 1, nex_a):
                    r = cre_a[jex]
                    s = anh_a[jex]
                    det_a[iex, jex] = G0a[p, s]
                    det_a[jex, iex] = G0a[r, q]
            ref[iw, idet] = xp.linalg.det(det_a)
    from ipie.utils.misc import dotdict

    empty = [[0], [0], [0], [0], [0]]
    trial = dotdict(
        {
            "cre_ex_a": empty + [xp.array([cre_a] * ndets, dtype=int)],
            "anh_ex_a": empty + [xp.array([anh_a] * ndets, dtype=int)],
            "cre_ex_b": empty + [xp.array([cre_a] * ndets, dtype=int)],
            "anh_ex_b": empty + [xp.array([anh_a] * ndets, dtype=int)],
            "occ_map_b": xp.arange(nmo, dtype=xp.int32),
            "occ_map_a": xp.arange(nmo, dtype=xp.int32),
            "nfrozen": 0,
        }
    )
    dets = xp.zeros_like(ref)
    det_mat_buffer = xp.zeros((nwalker, ndets, nex_a, nex_a),
            dtype=xp.complex128)
    wk.get_dets_nfold(
            trial.cre_ex_a[nex_a],
            trial.anh_ex_a[nex_a],
            trial.occ_map_a,
            0,
            G0,
            det_mat_buffer,
            dets
            )
    assert xp.allclose(ref, dets)

if __name__ == '__main__':
    test_get_dets_single_excitation_batched()
    test_get_dets_double_excitation_batched()
    test_get_dets_triple_excitation_batched()
    test_get_dets_nfold_excitation_batched()
