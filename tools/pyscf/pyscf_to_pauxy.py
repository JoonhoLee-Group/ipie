import argparse
import functools
import numpy
import h5py
import sys
from pauxy.utils.from_pyscf import dump_pauxy
from pauxy.utils.io import write_input

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
    parser.add_argument('-i', '--input', dest='input_scf', type=str,
                        default=None, help='PYSCF scf chkfile.')
    parser.add_argument('-o', '--output', dest='output', type=str,
                        default='afqmc.h5', help='Output file Hamiltonian.')
    parser.add_argument('-w', '--wavefile', dest='wfn', type=str,
                        default='afqmc.h5', help='Output file name for qmcpack trial.')
    parser.add_argument('-t', '--thresh', dest='thresh', type=float,
                        default=1e-5, help='Cholesky convergence threshold.')
    parser.add_argument('-s', '--sparse', dest='sparse', action='store_true',
                        default=False, help='Write in sparse format.')
    parser.add_argument('-sz', '--sparse_zero', dest='sparse_zero', type=float,
                        default=1e-16, help='Sparsity threshold')
    parser.add_argument('-b', '--back-prop', dest='bp', action='store_true',
                        default=False, help='Add back propagation option to json'
                        'input file.')
    parser.add_argument('-j', '--json-input', dest='json_input', type=str,
                        default='input.json', help='Name of input file.')
    parser.add_argument('-oao', '--oao', dest='oao', type=int,
                        default=1, help='whether to do oao')
    parser.add_argument('-ao', '--ao', dest='ao', type=int,
                        default=0, help='whether to do ao')

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
    dump_pauxy(chkfile=options.input_scf, hamil_file=options.output,
               wfn_file=options.wfn, chol_cut=options.thresh,
               sparse=options.sparse, sparse_zero=options.sparse_zero, ortho_ao=options.oao, ao=options.ao)
    write_input(options.json_input, options.output, options.wfn, options.bp)

if __name__ == '__main__':

    main(sys.argv[1:])

