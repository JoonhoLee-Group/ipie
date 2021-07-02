#!/usr/bin/env python
'''Simple analysis of PAUXY QMC output files.

By default data will be aggregated into a single output file with analysed_
prefixed to input filename.
'''
import argparse
import os
import sys
import pandas as pd
import numpy
from numpy import linalg
import json
from pauxy.analysis.blocking import reblock_rdm
_script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(_script_dir, 'analysis'))
from pauxy.analysis.blocking import analyse_estimates
import glob


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
    parser.add_argument('-s', '--start', type=int, dest='start_time',
                        default=1, help='Number of Samples to Skip Default: 1')
    parser.add_argument('-l', '--multi-sim', action='store_true',
                        dest='multi_sim', default=False,
                        help='Average over multiple simulations. By default '
                        'an attempt is made to group results by features.')
    parser.add_argument('-f', nargs='+', dest='filenames',
                        help='Space-separated list of files to analyse.')

    options = parser.parse_args(args)

    if not options.filenames:
        parser.print_help()
        sys.exit(1)

    return options


def main(args):
    """Run reblocking and data analysis on PAUXY output.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    None.
    """

    options = parse_args(args)
    if '*' in options.filenames[0]:
        files = glob.glob(options.filenames[0])
    else:
        files = options.filenames
    ordm, ordm_err = reblock_rdm(files, skip=options.start_time, est_type='back_propagated', rdm_type='one_rdm')
    nk = (ordm[0]+ordm[1]).diagonal()
    print("nk = {}".format(nk))
    Psym = ordm[0] + ordm[1]
    Psym = (Psym + numpy.conj(numpy.transpose(Psym))) * 0.5
    w, v = linalg.eig (Psym)
    print("eigval = {}".format(w.real))
    

if __name__ == '__main__':

    main(sys.argv[1:])
