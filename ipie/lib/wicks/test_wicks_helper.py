import numpy as np
import pytest

try:
    from ipie.lib.wicks.wicks_helper import (compute_opdm, count_set_bits,
                                             decode_det, encode_det,
                                             encode_dets, get_excitation_level,
                                             get_ia, get_perm_ia,
                                             print_bitstring)
    no_wicks = False
except ImportError:
    no_wicks = True

@pytest.mark.wicks
@pytest.mark.skipif(no_wicks, reason="lib.wicks not found.")
@pytest.mark.parametrize(
        "test_input,expected",
        [
            (([], [0,1,2,3]), ['0b10101010', '0b0']),
            (([0,3], [0,1,2,3]), ['0b11101011', '0b0']),
            (([1], [0,1,2,3]), ['0b10101110', '0b0']),
            (([], [0,1]), ['0b1010', '0b0']),
            (([1,35], [0,34]), ['0b110', '0b1100000'])
        ]
            )
def test_encode_det(test_input, expected):
    a, b = test_input
    a = np.array(a, dtype=np.int32)
    b = np.array(b, dtype=np.int32)
    det = encode_det(a, b)
    for i in range(len(det)):
        # print(i, bin(det[i]), expected[i])
        # # print(i, bin(det[i]), expected[i])
        assert bin(det[i]) == expected[i]

@pytest.mark.wicks
@pytest.mark.skipif(no_wicks, reason="lib.wicks not found.")
def test_encode_dets():
    occsa = np.array(
            [[0,1,2,3], [0,1,2,4]],
            dtype=np.int32
            )
    occsb = np.array(
            [[0, 1, 2, 3], [0,1,2,3]],
            dtype=np.int32
            )
    dets = encode_dets(occsa, occsb)
    for i in range(len(occsa)):
        d = encode_det(occsa[i], occsb[i])
        assert (dets[i] == d).all()



@pytest.mark.wicks
@pytest.mark.skipif(no_wicks, reason="lib.wicks not found.")
@pytest.mark.parametrize(
        "test_input,expected",
        [
            ([15,0], 4),
            ([37,0], 3),
            ([1001,0], 7)
        ]
            )
def test_count_set_bits(test_input, expected):
    nset = count_set_bits(np.array(test_input, dtype=np.uint64))
    assert nset == expected

@pytest.mark.wicks
@pytest.mark.skipif(no_wicks, reason="lib.wicks not found.")
@pytest.mark.parametrize(
        "test_input,expected",
        [
            # (([2,0], [1,0]), 1),
            # (([12,0], [3,0]), 2),
            (([1688832680394751, 2], [1125899906842623, 0]), 2)
        ]
            )
def test_get_excitation_level(test_input, expected):
    # print(bin(test_input[0]), bin(test_input[1]))
    # print(bin(test_input[0]^test_input[1]))
    nset = get_excitation_level(
            np.array(test_input[0], dtype=np.uint64),
            np.array(test_input[1], dtype=np.uint64)
            )
    assert nset == expected

@pytest.mark.wicks
@pytest.mark.skipif(no_wicks, reason="lib.wicks not found.")
@pytest.mark.parametrize(
        "test_input,expected",
        [
            (np.array([12,0], dtype=np.uint64), [2,3]),
            (np.array([5,0], dtype=np.uint64), [0,2]),
        ]
            )
def test_decode_det(test_input, expected):
    out = np.zeros(len(expected), dtype=np.int32)
    # print(out.size, out.shape, expected, np.array(expected).shape)
    decode_det(test_input, out)
    assert (out == expected).all()


@pytest.mark.wicks
@pytest.mark.skipif(no_wicks, reason="lib.wicks not found.")
def test_get_ia():
    # <1100|0^3|1010>
    ket = encode_det(
            np.array([], dtype=np.int32),
            np.array([0,1], dtype=np.int32)
            )
    bra = encode_det(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32)
            )
    ia = np.zeros(2, dtype=np.int32)
    get_ia(bra, ket, ia)
    assert (ia == [3, 0]).all()

@pytest.mark.wicks
@pytest.mark.skipif(no_wicks, reason="lib.wicks not found.")
@pytest.mark.parametrize(
        "test_input,expected",
        [
            ([[], [0,1], [3,0]], -1),
            ([[0,14//2], [0,9//2], [14, 8]], -1),
            ([[0,4//2], [1,13//2], [13, 17]], 1),
        ]
            )
def test_get_perm_ia(test_input, expected):
    # perm(0^3|0101>) = -1
    # |0101> = 1010 (binary)
    o1, o2, ia = test_input
    ket = encode_det(
            np.array(o1, dtype=np.int32),
            np.array(o2, dtype=np.int32)
            )
    print(bin(ket[0]), o1, o2, ia, 2*np.array(ia))
    perm = get_perm_ia(ket, ia[0], ia[1])
    assert perm == expected

@pytest.mark.wicks
@pytest.mark.skipif(no_wicks, reason="lib.wicks not found.")
def test_get_perm_ia_long():
    # perm(0^3|0101>) = -1
    # |0101> = 1010 (binary)
    ket = np.array([3236962232172543, 0], dtype=np.uint64)
    out = np.zeros(50, dtype=np.int32)
    decode_det(ket, out)
    print(out)
    print(bin(ket[0]), 51, 47)
    perm = get_perm_ia(ket, 51, 47)
    assert perm == 1

@pytest.mark.wicks
@pytest.mark.skipif(no_wicks, reason="lib.wicks not found.")
def test_compute_opdm():
    # perm(0^3|0101>) = -1
    # |0101> = 1010 (binary)
    coeffs = np.array([0.55, 0.1333, 0.001, 0.44])
    dets = []
    dets.append(encode_det(
            np.array([0], dtype=np.int32),
            np.array([0], dtype=np.int32)
            ))
    dets.append(encode_det(
            np.array([0], dtype=np.int32),
            np.array([1], dtype=np.int32)
            ))
    dets.append(encode_det(
            np.array([1], dtype=np.int32),
            np.array([0], dtype=np.int32)
            ))
    dets.append(encode_det(
            np.array([1], dtype=np.int32),
            np.array([1], dtype=np.int32)
            ))
    dets = np.array(dets, dtype=np.ulonglong)
    P = compute_opdm(
            coeffs,
            dets,
            4,
            2)
