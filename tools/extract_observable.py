#!/usr/bin/env python
'''Exctact element of green's function'''

import argparse
import sys
import numpy
import os
import pandas as pd
import json
_script_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.join(_script_dir, 'analysis'))
import matplotlib.pyplot as plt
# from pauxy.analysis.extraction import analysed_itcf
# from pauxy.analysis.extraction import analysed_energies, extract_hdf5_simple
from pauxy.analysis.extraction import (
        extract_mixed_estimates,
        get_metadata
        )
import matplotlib.pyplot as pl


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
    parser.add_argument('-s', '--spin', type=str, dest='spin',
                        default=None, help='Spin component to extract.'
                        'Options: up/down')
    parser.add_argument('-t', '--type', type=str, dest='type',
                        default=None, help='Type of green\'s function to extract.'
                        'Options: lesser/greater')
    parser.add_argument('-k', '--kspace', dest='kspace', action='store_true',
                        default=False, help='Extract kspace green\'s function.')
    parser.add_argument('-e', '--elements',
                        type=lambda s: [int(item) for item in s.split(',')],
                        dest='elements', default=None,
                        help='Element to extract.')
    parser.add_argument('-o', '--observable', type=str, dest='obs',
                        default='None', help='Data to extract')
    parser.add_argument('-p', '--plot-energy', action='store_true', dest='plot',
                        default=False, help='Plot energy trace.')
    parser.add_argument('-f', nargs='+', dest='filename',
                        help='Space-separated list of files to analyse.')

    options = parser.parse_args(args)

    if not options.filename:
        parser.print_help()
        sys.exit(1)

    return options


def main(args):
    """Extract observable from analysed output.

    Parameters
    ----------
    args : list of strings
        command-line arguments.

    Returns
    -------
    results : :class:`pandas.DataFrame`
        Anysed results.
    """

    options = parse_args(args)
    print_index = False
    if options.obs == 'itcf':
        results = analysed_itcf(options.filename[0], options.elements,
                                options.spin, options.type, options.kspace)
    elif options.obs == 'energy':
        results = analysed_energies(options.filename[0], 'mixed')
    elif options.obs == 'back_propagated':
        results = analysed_energies(options.filename[0], 'back_propagated')
    elif 'correlation' in options.obs:
        ctype = options.obs.replace('_correlation', '')
        results = correlation_function(options.filename[0],
                                       ctype,
                                       options.elements)
        print_index = True
    elif options.plot:
        data = extract_mixed_estimates(options.filename[0])
        md = get_metadata(options.filename[0])
        fp = md['propagators']['free_projection']
        dt = md['qmc']['dt']
        mc = md['qmc']['nsteps']
        data = data[abs(data.Weight) > 0.0]
        tau = numpy.arange(0,len(data)) * mc * dt
        if fp:
            pl.plot(tau, numpy.real(data.ENumer/data.EDenom))
            pl.xlabel(r"$\tau$ (au)")
            pl.ylabel(r"Energy (au)")
            pl.show()
        else:
            pl.plot(tau, data[options.obs].real)
            pl.xlabel(r"$\tau$ (au)")
            pl.ylabel(r"{} (au)".format(options.obs))
            pl.show()
    else:
        print ('Unknown observable')

    if not options.plot:
        print (results.to_string())
        results.to_csv("%s"%options.obs)

if __name__ == '__main__':

    main(sys.argv[1:])
