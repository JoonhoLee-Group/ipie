#!/usr/bin/env python

import sys
import argparse
import pprint

import json
import glob
import numpy
import scipy.optimize
import pandas as pd

from ipie.analysis.blocking import (
        average_ratio
        )
from ipie.analysis.extraction import (
        extract_observable,
        get_metadata, 
        get_sys_param
        )

from ipie.addons.thermal.analysis.extraction import set_info


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


def analyse(files, block_idx=1):
    sims = []
    files = sorted(files)

    for f in files:
        data_energy = extract_observable(f, name='energy', block_idx=block_idx)
        data_nav = extract_observable(f, name='nav', block_idx=block_idx)
        data = pd.concat([data_energy, data_nav['Nav']], axis=1)
        md = get_metadata(f)
        keys = set_info(data, md)
        sims.append(data[1:])

    full = pd.concat(sims).groupby(keys, sort=False)

    analysed = []
    for i, g in full:
        cols = ["ETotal", "E1Body", "E2Body", "Nav"]
        averaged = pd.DataFrame(index=[0])

        for c in cols:
            mean = numpy.real(g[c].values).mean()
            error = scipy.stats.sem(numpy.real(g[c].values), ddof=1)
            averaged[c] = [mean]
            averaged[c + "_error"] = [error]

        for k, v in zip(full.keys, i):
            averaged[k] = v

        analysed.append(averaged)

    return pd.concat(analysed).reset_index(drop=True).sort_values(by=keys)


def nav_mu(mu, coeffs):
    return numpy.polyval(coeffs, mu)


def find_chem_pot(data, target, vol, order=3, plot=False):
    print(f"# System volume: {vol}.")
    print(f"# Target number of electrons: {vol * target}.")
    nav = data.Nav.values / vol
    nav_error = data.Nav_error.values / vol
    # Half filling special case where error bar is zero.
    zeros = numpy.where(nav_error == 0)[0]
    nav_error[zeros] = 1e-8
    mus = data.mu.values
    delta = nav - target
    s = 0
    e = len(delta)
    rmin = None

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

    data = analyse(files)

    if options.fit_chem_pot:
        name = get_sys_param(files[0], 'name')
        vol = 1.
        mu = find_chem_pot(data, options.nav, vol,
                           order=options.order, plot=options.plot)

        if mu is not None:
            print("# Optimal chemical potential found to be: {}.".format(mu))

        else:
            print("# Failed to find chemical potential.")

    print(data.to_string(index=False))


if __name__ == '__main__':
    main(sys.argv[1:])
