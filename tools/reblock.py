#!/usr/bin/env python
'''Simple analysis of PAUXY QMC output files.

By default data will be aggregated into a single output file with analysed_
prefixed to input filename.
'''
import argparse
import os
import sys
import pandas as pd
import json
_script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(_script_dir, 'analysis'))
from pauxy.analysis.blocking import (
        analyse_estimates,
        average_fp
        )
from pauxy.analysis.extraction import extract_data_sets
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
    parser.add_argument('-s', '--start', type=float, dest='start_time',
                        default=0, help='Imaginary time in a.u. after which we '
                        'gather statistics.  Default: 0')
    parser.add_argument('-m', '--multi-sim', action='store_true',
                        dest='multi_sim', default=False,
                        help='Average over multiple simulations. By default '
                        'an attempt is made to group results by features.')
    parser.add_argument('--free-proj', action='store_true', default=False,
                        dest='free_proj',
                        help='Free projection error analysis using mulitple'
                        ' simulations')
    parser.add_argument('-t', '--average-tau', action='store_true',
                        dest='av_tau', default=False,
                        help='Compute average of energy as a function of tau.')
    parser.add_argument('-c', '--correlation', dest='cfunc', action='store_true',
                        default=False, help='Extract correlation functions.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        default=False, help='Analyse all all energy components.'
                        ' Default False.')
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
    if options.free_proj:
        data = extract_data_sets(files, 'basic', 'energies')
        res = average_fp(data)
        print(res.to_string(index=False))
    else:
        analyse_estimates(files, start_time=options.start_time,
                          multi_sim=options.multi_sim, av_tau=options.av_tau,
                          verbose=options.verbose)
if __name__ == '__main__':
    main(sys.argv[1:])
