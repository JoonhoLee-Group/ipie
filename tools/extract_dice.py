import argparse
import numpy as np
import struct
import sys

from ipie.utils.io import write_qmcpack_wfn

def parse_args(args):
    """Parse command-line arguments.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    options : :class:`argparse.ArgumentParser`
        Command line arguments.
    """

    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument('--nalpha', type=int, dest='nalpha',
                        default=0, help='Total number of alpha electrons.')
    parser.add_argument('--nbeta', type=int, dest='nbeta',
                        default=0, help='Total number of beta electrons.')
    parser.add_argument('--nmo', type=int, dest='nmo',
                        default=0, help='Total number of MOs')
    parser.add_argument('--nfrozen', type=int, dest='nfrozen',
                        default=0, help='Number of frozen core orbitals to'
                        ' reinclude.')
    parser.add_argument('--ndets', type=int, dest='ndets',
                        default=-1, help='Number of frozen core orbitals to'
                        ' reinclude.')
    parser.add_argument('--verbose', dest='verbose', action='store_true',
                        default=False, help='Number of frozen core orbitals to'
                        ' reinclude.')

    options = parser.parse_args(args)

    if (options.nalpha == 0 and options.nbeta == 0) or options.nmo == 0:
        parser.print_help()
        sys.exit(1)

    return options


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

def read_dice_wavefunction(filename, ndets=None):
    print("Reading Dice wavefunction from dets.bin")
    with open(filename, 'rb') as f:
        data = f.read()
    _chr = 1
    _int = 4
    _dou = 8
    ndets_in_file = struct.unpack('<I', data[:4])[0]
    norbs = struct.unpack('<I', data[4:8])[0]
    wfn_data = data[8:]
    coeffs = []
    occs = []
    start = 0
    print(f"Number of determinants specified: {ndets}")
    print(f"Number of determinants in dets.bin : {ndets_in_file}")
    print(f"Number of orbitals : {norbs}")
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

def convert_phase(coeff0, occa_ref, occb_ref, verbose=False):
    print("Converting phase to account for abab -> aabb")
    ndets = len(coeff0)
    coeffs = np.zeros(len(coeff0), dtype=np.complex128)
    for i in range(ndets):
        if verbose and i % (int(0.1*ndets)) == 0 and i > 0:
            done = float(i)/ndets
            print(f"convert phase {i}. Percent: {done}")
        doubles = list(set(occa_ref[i])&set(occb_ref[i]))
        occa0 = np.array(occa_ref[i])
        occb0 = np.array(occb_ref[i])

        count = 0
        for ocb in occb0:
            passing_alpha = np.where(occa0 > ocb)[0]
            count += len(passing_alpha)

        phase = (-1)**count
        coeffs[i] = coeffs0[i]
    ixs = np.argsort(np.abs(coeffs))[::-1]
    coeffs = coeffs[ixs]
    occa = np.array(occa_ref)[ixs]
    occb = np.array(occb_ref)[ixs]

    return coeffs, occa, occb

if __name__ == '__main__':
    options = parse_args(sys.argv[1:])
    for key, val in vars(options).items():
        print(f"{key:<10s} : {val:>10d}")
    if options.ndets > 0:
        ndets = options.ndets
    else:
        ndets = None
    coeffs0, occa0, occb0 = read_dice_wavefunction('dets.bin', ndets=ndets)
    coeffs, occa, occb = convert_phase(coeffs0, occa0, occb0, verbose=options.verbose)
    if options.nfrozen > 0:
        core = [i for i in range(options.nfrozen)]
        occa = [np.array(core + [orb + options.nfrozen for orb in oa], dtype=np.int32) for oa in occa]
        occb = [np.array(core + [orb + options.nfrozen for orb in ob], dtype=np.int32) for ob in occb]
    write_qmcpack_wfn(
            'wfn.h5',
            (coeffs, occa, occb),
            'UHF',
            (options.nalpha, options.nbeta),
            options.nmo)
