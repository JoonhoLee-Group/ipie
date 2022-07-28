import argparse
import struct
import sys

import numpy as np

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

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nalpha",
        type=int,
        dest="nalpha",
        default=0,
        help="Total number of alpha electrons. If nfrozen > 0 it should be "
        "nfrozen + number of electrons in active space.",
    )
    parser.add_argument(
        "--nbeta",
        type=int,
        dest="nbeta",
        default=0,
        help="Total number of beta electrons. If nfrozen > 0 it should be "
        "nfrozen + number of electrons in active space.",
    )
    parser.add_argument(
        "--nmo", type=int, dest="nmo", default=0, help="Total number of MOs"
    )
    parser.add_argument(
        "--dice-wfn",
        type=str,
        dest="dice_file",
        default="dets.bin",
        help="Wavefunction file containing dice determinants. Default: dets.bin.",
    )
    parser.add_argument(
        "--filename",
        type=str,
        dest="filename",
        default="wfn.h5",
        help="Wavefunction file for AFQMC MSD. Default: dets.bin.",
    )
    parser.add_argument(
        "--nfrozen",
        type=int,
        dest="nfrozen",
        default=0,
        help="Number of frozen core orbitals to" " reinclude.",
    )
    parser.add_argument(
        "--sort",
        dest="sort",
        action="store_true",
        help="Sort wavefunction by ci coefficient magnitude.",
    )
    parser.add_argument(
        "--ndets",
        type=int,
        dest="ndets",
        default=-1,
        help="Number of frozen core orbitals to" " reinclude.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Number of frozen core orbitals to" " reinclude.",
    )

    options = parser.parse_args(args)

    if (options.nalpha == 0 and options.nbeta == 0) or options.nmo == 0:
        parser.print_help()
        sys.exit(1)

    return options


def decode_dice_det(occs):
    occ_a = []
    occ_b = []
    for i, occ in enumerate(occs):
        if occ == "2":
            occ_a.append(i)
            occ_b.append(i)
        elif occ == "a":
            occ_a.append(i)
        elif occ == "b":
            occ_b.append(i)
    return occ_a, occ_b


def read_dice_wavefunction(filename):
    print("Reading Dice wavefunction from dets.bin")
    with open(filename, "rb") as f:
        data = f.read()
    _chr = 1
    _int = 4
    _dou = 8
    ndets_in_file = struct.unpack("<I", data[:4])[0]
    norbs = struct.unpack("<I", data[4:8])[0]
    wfn_data = data[8:]
    coeffs = []
    occs = []
    start = 0
    print(f"Number of determinants in dets.bin : {ndets_in_file}")
    print(f"Number of orbitals : {norbs}")
    for idet in range(ndets_in_file):
        coeff = struct.unpack("<d", wfn_data[start : start + _dou])[0]
        coeffs.append(coeff)
        start += _dou
        occ_i = wfn_data[start : start + norbs]
        occ_lists = decode_dice_det(str(occ_i)[2:-1])
        occs.append(occ_lists)
        start += norbs
    print("Finished reading wavefunction from file.")
    oa, ob = zip(*occs)
    return np.array(coeffs, dtype=np.complex128), np.array(oa, dtype=np.int32), np.array(ob, dtype=np.int32)


def convert_phase(coeff0, occa_ref, occb_ref, verbose=False):
    print("Converting phase to account for abab -> aabb")
    ndets = len(coeff0)
    coeffs = np.zeros(len(coeff0), dtype=np.complex128)
    for i in range(ndets):
        if verbose and i % (int(0.1 * ndets)) == 0 and i > 0:
            done = float(i) / ndets
            print(f"convert phase {i}. Percent: {done}")
        doubles = list(set(occa_ref[i]) & set(occb_ref[i]))
        occa0 = np.array(occa_ref[i])
        occb0 = np.array(occb_ref[i])

        count = 0
        for ocb in occb0:
            passing_alpha = np.where(occa0 > ocb)[0]
            count += len(passing_alpha)

        phase = (-1) ** count
        coeffs[i] = coeffs0[i]
    ixs = np.argsort(np.abs(coeffs))[::-1]
    coeffs = coeffs[ixs]
    occa = np.array(occa_ref)[ixs]
    occb = np.array(occb_ref)[ixs]

    return coeffs, occa, occb


if __name__ == "__main__":
    options = parse_args(sys.argv[1:])
    for key, val in vars(options).items():
        if isinstance(val, str):
            print(f"{key:<10s} : {val:>10s}")
        else:
            print(f"{key:<10s} : {val:>10d}")
    coeffs0, occa0, occb0 = read_dice_wavefunction(options.dice_file)
    if options.sort:
        ix = np.argsort(np.abs(coeffs0))[::-1]
        coeffs0 = coeffs0[ix]
        occa0 = occa0[ix]
        occb0 = occb0[ix]
    if options.ndets > -1:
        print(f"Number of determinants specified: {options.ndets}")
        coeffs0 = coeffs0[:options.ndets]
        occa0 = occa0[:options.ndets]
        occb0 = occb0[:options.ndets]
    coeffs, occa, occb = convert_phase(coeffs0, occa0, occb0, verbose=options.verbose)
    if options.nfrozen > 0:
        if options.verbose:
            print(f"Reinserting {options.nfrozen} frozen core orbitals")
        core = [i for i in range(options.nfrozen)]
        occa = [
            np.array(core + [orb + options.nfrozen for orb in oa], dtype=np.int32)
            for oa in occa
        ]
        occb = [
            np.array(core + [orb + options.nfrozen for orb in ob], dtype=np.int32)
            for ob in occb
        ]
    write_qmcpack_wfn(
        options.filename,
        (coeffs, occa, occb),
        "UHF",
        (options.nalpha, options.nbeta),
        options.nmo,
    )
