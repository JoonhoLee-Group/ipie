# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee
#

#!/usr/bin/env python

import argparse
import glob
import sys

import numpy
import pandas as pd
import scipy.optimize

from ipie.addons.thermal.analysis.extraction import set_info
from ipie.analysis.extraction import extract_observable, get_metadata


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
        "-f", nargs="+", dest="filenames", help="Space-separated list of files to analyse."
    )

    options = parser.parse_args(args)

    if not options.filenames:
        parser.print_help()
        sys.exit(1)

    return options


def analyse(files, block_idx=1):
    sims = []
    files = sorted(files)

    for f in files:
        data_energy = extract_observable(f, name="energy", block_idx=block_idx)
        data_nav = extract_observable(f, name="nav", block_idx=block_idx)
        data = pd.concat([data_energy, data_nav["Nav"]], axis=1)
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
    if "*" in options.filenames[0]:
        files = glob.glob(options.filenames[0])

    else:
        files = options.filenames

    data = analyse(files)

    print(data.to_string(index=False))


if __name__ == "__main__":
    main(sys.argv[1:])
