import struct
import numpy as np

def decode_dice_det(occs):
    occ_a = []
    occ_b = []
    for i, occ in enumerate(occs):
        if occ == '2':
            occ_a.append(i)
            occ_b.append(i)
        elif occ == 'a':
            occ_a.append(i)
        elif occ == 'b':
            occ_b.append(i)
    return occ_a, occ_b

def read_dice_wavefunction(filename):
    with open(filename, 'rb') as f:
        data = f.read()
    _chr = 1
    _int = 4
    _dou = 8
    ndets = struct.unpack('<I', data[:4])[0]
    norbs = struct.unpack('<I', data[4:8])[0]
    wfn_data = data[8:]
    coeffs = []
    occs = []
    start = 0
    for idet in range(ndets):
        coeff = struct.unpack('<d', wfn_data[start:start+_dou])[0]
        coeffs.append(coeff)
        start += _dou
        occ_i = wfn_data[start:start+norbs]
        occ_lists = decode_dice_det(str(occ_i)[2:-1])
        occs.append(occ_lists)
        start += norbs
    oa, ob = zip(*occs)
    return np.array(coeffs, dtype=np.complex128), oa, ob

if __name__ == '__main__':
    coeffs, occa, occb = read_dice_wavefunction('dets.bin')
    from pie.utils.io import write_qmcpack_wfn
    write_qmcpack_wfn('wfn.h5', (coeffs, occa, occb), 'UHF', (25, 25), 50)
