import argparse
import sys

import numpy as np

from ipie.utils.from_dice import convert_phase, read_dice_wavefunction
from ipie.utils.io import write_wavefunction


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
        help="Wavefunction file for AFQMC MSD. Default: wfn.h5.",
    )
    parser.add_argument(
        "--sort",
        dest="sort",
        action="store_true",
        help="Sort wavefunction by ci coefficient magnitude.",
    )
    parser.add_argument(
        "--convert-phase",
        dest="convert_phase",
        action="store_true",
        help="Convert phase necessary if determinants are stored in abab order rather than aabb.",
    )
    parser.add_argument(
        "--ndets",
        type=int,
        dest="ndets",
        default=-1,
        help="Number of determinants to write.",
    )
    parser.add_argument(
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Verbose output.",
    )

    options = parser.parse_args(args)

    return options


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
        coeffs0 = coeffs0[: options.ndets]
        occa0 = occa0[: options.ndets]
        occb0 = occb0[: options.ndets]
    if options.convert_phase:
        coeffs0, occa0, occb0 = convert_phase(coeffs0, occa0, occb0, verbose=options.verbose)
    write_wavefunction(
        (coeffs0, occa0, occb0),
        filename=options.filename,
    )
