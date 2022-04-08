import os
import numpy as np
import ctypes
from numpy.ctypeslib import ndpointer

_path = os.path.dirname(__file__)
_wicks_helper = np.ctypeslib.load_library('libwicks_helper', _path)
DET_LEN = 2

def encode_dets(occsa, occsb):
    assert isinstance(occsa, np.ndarray)
    assert isinstance(occsb, np.ndarray)
    assert len(occsa.shape) == 2
    assert len(occsb.shape) == 2
    ndets = occsa.shape[0]
    dets = np.zeros((ndets, DET_LEN), dtype=np.uint64)
    fun = _wicks_helper.encode_dets
    fun.restype = ctypes.c_ulonglong
    # print(occsa.shape)
    nocca = occsa.shape[1]
    noccb = occsa.shape[1]
    fun.argtypes = [ndpointer(shape=(ndets, nocca), dtype=ctypes.c_int, flags="C_CONTIGUOUS"),
                    ndpointer(shape=(ndets, noccb), dtype=ctypes.c_int, flags="C_CONTIGUOUS"),
                    ndpointer(shape=(ndets, DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                    ctypes.c_size_t,
                    ctypes.c_size_t,
                    ctypes.c_size_t
                    ]
    _wicks_helper.encode_dets(
            occsa,
            occsb,
            dets,
            nocca,
            noccb,
            ndets,
            )
    return dets

def encode_det(a, b):
    assert isinstance(a, np.ndarray)
    assert isinstance(b, np.ndarray)
    fun = _wicks_helper.encode_det
    fun.restype = None
    fun.argtypes = [ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                    ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                    ndpointer(shape=(DET_LEN,), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                    ctypes.c_size_t,
                    ctypes.c_size_t
                    ]
    out = np.zeros(DET_LEN, dtype=np.uint64)
    det = _wicks_helper.encode_det(
            a,
            b,
            out,
            a.size,
            b.size
            )
    return out

def count_set_bits(a):
    fun = _wicks_helper.count_set_bits
    fun.restype = ctypes.c_int
    fun.argtypes = [ndpointer(shape=(DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS")]
    nset = _wicks_helper.count_set_bits(
            a
            )
    return nset

def get_excitation_level(a, b):
    fun = _wicks_helper.get_excitation_level
    fun.restype = ctypes.c_int
    fun.argtypes = [
                    ndpointer(shape=(DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                    ndpointer(shape=(DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                    ]
    nexcit = _wicks_helper.get_excitation_level(
            a,
            b
            )
    return nexcit

def decode_det(det, occs):
    fun = _wicks_helper.decode_det
    fun.restype = None
    fun.argtypes = [
            ndpointer(shape=(DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
            ndpointer(shape=(occs.size), dtype=ctypes.c_int, flags="C_CONTIGUOUS"),
            ctypes.c_size_t
            ]
    print(det.size, det.shape, occs.size, occs.shape)
    _wicks_helper.decode_det(
            det,
            occs,
            occs.size
            )

def get_ia(det_bra, det_ket, ia):
    fun = _wicks_helper.get_ia
    fun.restype = None
    fun.argtypes = [
            ndpointer(shape=(DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
            ndpointer(shape=(DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
            ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
            ]
    _wicks_helper.get_ia(
            det_bra,
            det_ket,
            ia
            )

def get_perm_ia(det_ket, i, a):
    fun = _wicks_helper.get_perm_ia
    fun.restype = ctypes.c_int
    fun.argtypes = [
            ndpointer(shape=(DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
            ctypes.c_int,
            ctypes.c_int,
            ]
    # print(det_ket, type(det_ket), det_ket.dtype)
    perm = _wicks_helper.get_perm_ia(
            det_ket,
            i,
            a
            )
    return perm

def compute_opdm(
        ci_coeffs,
        dets,
        norbs,
        nelec):
    fun = _wicks_helper.compute_density_matrix
    fun.restype = None
    ndets = len(ci_coeffs)
    if ci_coeffs.dtype == np.complex128:
        fun.argtypes = [
                ndpointer(np.complex128, flags="C_CONTIGUOUS"),
                ndpointer(shape=(ndets, DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                ndpointer(np.complex128, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ]
    elif ci_coeffs.dtype == np.float64:
        fun.argtypes = [
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(shape=(ndets, DET_LEN), dtype=ctypes.c_ulonglong, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                ndpointer(ctypes.c_int, flags="C_CONTIGUOUS"),
                ctypes.c_size_t,
                ctypes.c_size_t,
                ctypes.c_size_t,
                ]
    else:
        raise TypeError("Unknown type for ci_coeffs")
    opdm = np.zeros((2, norbs, norbs), dtype=ci_coeffs.dtype)
    occs = np.zeros((nelec), dtype=np.int32)
    _wicks_helper.compute_density_matrix(
            ci_coeffs,
            dets,
            opdm,
            occs,
            ci_coeffs.size,
            norbs,
            nelec
            )
    return opdm

def convert_phase(occa, occb):
    ndet = len(occa)
    phases = np.zeros(ndet)
    for i in range(ndet):
        doubles = list(set(occa[i])&set(occb[i]))
        occa0 = np.array(occa[i])
        occb0 = np.array(occb[i])

        count = 0
        for ocb in occb0:
            passing_alpha = np.where(occa0 > ocb)[0]
            count += len(passing_alpha)

        phases[i] = (-1)**count

    return phases

def print_bitstring(bitstring, nbits=64):
    mask = np.uint64(1)
    out = ''
    for bs in bitstring:
        out += ''.join('1' if bs & (mask << np.uint64(i)) else '0' for i in range(nbits))
    return out[::-1]
