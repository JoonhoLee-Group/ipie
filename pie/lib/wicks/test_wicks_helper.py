import pytest
import numpy as np

from pie.lib.wicks.wicks_helper import (
        encode_det,
        encode_dets,
        count_set_bits,
        get_excitation_level,
        decode_det,
        get_ia,
        get_perm_ia,
        compute_opdm
        )

@pytest.mark.unit
@pytest.mark.parametrize(
        "test_input,expected",
        [
            (([], [0,1,2,3]), '0b10101010'),
            (([0,3], [0,1,2,3]), '0b11101011'),
            (([1], [0,1,2,3]), '0b10101110'),
            (([], [0,1]), '0b1010')
        ]
            )
def test_encode_det(test_input, expected):
    a, b = test_input
    a = np.array(a, dtype=np.int32)
    b = np.array(b, dtype=np.int32)
    det = encode_det(a, b)
    assert bin(det) == expected

@pytest.mark.unit
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
    for i in range(2):
        d = encode_det(occsa[i], occsb[i])
        assert dets[i] == d



@pytest.mark.unit
@pytest.mark.parametrize(
        "test_input,expected",
        [
            (15, 4),
            (37, 3),
            (1001, 7)
        ]
            )
def test_count_set_bits(test_input, expected):
    nset = count_set_bits(test_input)
    assert nset == expected

@pytest.mark.unit
@pytest.mark.parametrize(
        "test_input,expected",
        [
            ((2, 1), 1),
            ((12, 3), 2),
        ]
            )
def test_get_excitation_level(test_input, expected):
    # print(bin(test_input[0]), bin(test_input[1]))
    # print(bin(test_input[0]^test_input[1]))
    nset = get_excitation_level(test_input[0], test_input[1])
    assert nset == expected

@pytest.mark.unit
@pytest.mark.parametrize(
        "test_input,expected",
        [
            (12, [2,3]),
            (5, [0,2]),
        ]
            )
def test_decode_det(test_input, expected):
    out = np.zeros(len(expected), dtype=np.int32)
    decode_det(test_input, out)
    assert (out == expected).all()


@pytest.mark.unit
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

@pytest.mark.unit
def test_get_perm_ia():
    # perm(0^3|0101>) = -1
    # |0101> = 1010 (binary)
    ket = encode_det(
            np.array([], dtype=np.int32),
            np.array([0,1], dtype=np.int32)
            )
    perm = get_perm_ia(ket, 3, 0)
    assert perm == -1

@pytest.mark.unit
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
