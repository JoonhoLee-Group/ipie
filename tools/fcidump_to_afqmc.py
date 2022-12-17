
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
# Authors: Joonho Lee
#          Fionn Malone <fionn.malone@gmail.com>
#          Nick Rubin <rubinnc0@gmail.com>
#

#! /usr/bin/env python3

import argparse
import sys
import time

import numpy
import scipy.sparse

from ipie.utils.hamiltonian_converter import read_fcidump
from ipie.utils.io import write_qmcpack_dense
from ipie.utils.linalg import modified_cholesky


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
        dest="input_file",
        type=str,
        default=None,
        help="Input FCIDUMP file.",
    )
    parser.add_argument(
        "-o",
        "--output",
        dest="output_file",
        type=str,
        default="fcidump.h5",
        help="Output file name for ipie data.",
    )
    parser.add_argument(
        "--write-complex",
        dest="write_complex",
        action="store_true",
        default=False,
        help="Output integrals in complex format.",
    )
    parser.add_argument(
        "-t",
        "--cholesky-threshold",
        dest="thresh",
        type=float,
        default=1e-5,
        help="Cholesky convergence threshold.",
    )
    parser.add_argument(
        "-s",
        "--symmetry",
        dest="symm",
        type=int,
        default=8,
        help="Symmetry of integral file (1,4,8).",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Verbose output.",
    )

    options = parser.parse_args(args)

    if not options.input_file:
        parser.print_help()
        sys.exit(1)

    return options


def main(args):
    """Convert FCIDUMP to QMCPACK readable Hamiltonian format.

    Parameters
    ----------
    args : list of strings
        command-line arguments.
    """
    options = parse_args(args)
    (hcore, eri, ecore, nelec) = read_fcidump(
        options.input_file, symmetry=options.symm, verbose=options.verbose
    )
    norb = hcore.shape[-1]

    # If the ERIs are complex then we need to form M_{(ik),(lj}} which is
    # Hermitian. For real integrals we will have 8-fold symmetry so trasposing
    # will have no effect.
    eri = numpy.transpose(eri, (0, 1, 3, 2))

    chol = modified_cholesky(
        eri.reshape(norb**2, norb**2), options.thresh, options.verbose, cmax=20
    ).T.copy()
    cplx_chol = options.write_complex or numpy.any(abs(eri.imag) > 1e-14)
    write_qmcpack_dense(
        hcore,
        chol,
        nelec,
        norb,
        enuc=ecore,
        real_chol=(not cplx_chol),
        filename=options.output_file,
    )


if __name__ == "__main__":
    main(sys.argv[1:])
