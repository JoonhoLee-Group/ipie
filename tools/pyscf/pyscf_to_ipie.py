import argparse
import functools
import sys

import h5py
import numpy

from ipie.utils.from_pyscf import (
        gen_ipie_input_from_pyscf_chk,
        load_from_pyscf_chkfile
        )
from ipie.utils.io import write_json_input_file



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
        "-i",
        "--input",
        dest="input_scf",
        type=str,
        default=None,
        help="PYSCF scf chkfile.",
    )
    parser.add_argument(
        "--hamiltonian",
        dest="output",
        type=str,
        default="hamiltonian.h5",
        help="Output file Hamiltonian.",
    )
    parser.add_argument(
        "--wavefunction",
        dest="wfn",
        type=str,
        default="wavefunction.h5",
        help="Output file name for qmcpack trial.",
    )
    parser.add_argument(
        "-e",
        "--estimates",
        dest="est",
        type=str,
        default="estimates.0.h5",
        help="Output file name for estimates.",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        dest="thresh",
        type=float,
        default=1e-5,
        help="Cholesky convergence threshold.",
    )
    parser.add_argument(
        "-j",
        "--json-input",
        dest="json_input",
        type=str,
        default="input.json",
        help="Name of input file.",
    )
    parser.add_argument(
        "--mcscf",
        dest="mcscf",
        action="store_true",
        default=False,
        help="Use mcscf input to generate multi-slater trial wavefunction.",
    )
    parser.add_argument(
        "--frozen-core",
        dest="num_frozen_core",
        type=int,
        default=0,
        help="Number of core orbitals to freeze.",
    )
    parser.add_argument(
        "-o", "--ortho-ao", dest="oao", action="store_true", help="Whether to do"
        " use orthogonalized AO basis."
    )
    parser.add_argument(
        "--lin-dep", dest="lin_dep", type=float, default=0, help="Linear "
        "dependency threshold for canonical orthogonalization."
    )
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", help="Verbose output."
    )

    options = parser.parse_args(args)

    if not options.input_scf:
        parser.print_help()
        sys.exit(1)

    return options


def main(args):
    """Extract observable from analysed output.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """

    options = parse_args(args)
    gen_ipie_input_from_pyscf_chk(
        options.input_scf,
        hamil_file=options.output,
        wfn_file=options.wfn,
        verbose=options.verbose,
        chol_cut=options.thresh,
        ortho_ao=options.oao,
        mcscf=options.mcscf,
        num_frozen_core=options.num_frozen_core,
        linear_dep_thresh=options.lin_dep,
    )
    scf_data = load_from_pyscf_chkfile(options.input_scf)
    nelec_mol = scf_data['mol'].nelec
    nfzn = options.num_frozen_core
    nelec_sim = (nelec_mol[0]-nfzn, nelec_mol[1]-nfzn)
    write_json_input_file(
        options.json_input, options.output, options.wfn, nelec_sim,
        estimates_filename=options.est,
    )


if __name__ == "__main__":

    main(sys.argv[1:])
