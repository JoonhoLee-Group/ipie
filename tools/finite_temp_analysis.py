#!/usr/bin/env python

import glob
import numpy
try:
    import matplotlib.pyplot as pl
except ImportError:
    pass
import sys
from pauxy.analysis.thermal import analyse_energy, find_chem_pot
from pauxy.analysis.extraction import get_sys_param
import scipy.optimize

import argparse
import os
import sys
import pandas as pd
import json
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
    parser.add_argument('-c', '--chem-pot', dest='fit_chem_pot',
                        action='store_true', default=False,
                        help='Estimate optimal chemical potential')
    parser.add_argument('-n', '--nav', dest='nav', type=float,
                        help='Target electron density.')
    parser.add_argument('-o', '--order', dest='order', type=int,
                        default=3, help='Order polynomial to fit.')
    parser.add_argument('-p', '--plot', dest='plot', action='store_true',
                        help='Plot density vs. mu.')
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
    data = analyse_energy(files)
    if options.fit_chem_pot:
        # Assuming all files have same
        vol = get_sys_param(files[0], 'vol')
        name = get_sys_param(files[0], 'name')
        if name == 'UEG' or name == 'Generic':
            vol = 1.0
        mu = find_chem_pot(data, options.nav, vol,
                           order=options.order, plot=options.plot)
        if mu is not None:
            print("# Optimal chemical potential found to be: {}.".format(mu))
        else:
            print("# Failed to find chemical potential.")

    print(data.to_string(index=False))

if __name__ == '__main__':

    main(sys.argv[1:])
