
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
# Authors: Joonho Lee <linusjoonho@gmail.com>
#          Fionn Malone <fionn.malone@gmail.com>
#

import numpy
import pandas as pd

# Stolen from https://dfm.io/posts/autocorr/


def next_pow_two(n):
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(x, norm=True):
    x = numpy.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = numpy.fft.fft(x - numpy.mean(x), n=2 * n)
    acf = numpy.fft.ifft(f * numpy.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


# Automated windowing procedure following Sokal (1989)
def auto_window(taus, c):
    m = numpy.arange(len(taus)) < c * taus
    if numpy.any(m):
        return numpy.argmin(m)
    return len(taus) - 1


# Following the suggestion from Goodman & Weare (2010)
def autocorr_gw2010(y, c=5.0):
    f = autocorr_func_1d(y)
    taus = 2.0 * numpy.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def reblock_by_autocorr(y, name="ETotal", verbose=False):
    """Perform error analysis on ipie data using autocorrelation function

    Parameters
    ----------
    y : pd.DataFrame
        Output of of QMC calculation (mixed estimates). 
    name : string
        Which column to "reblock".
    verbose : bool
        Print information about statistics to stdout.

    Returns
    -------
    df : pd.DataFrame 
        Analysed data with errorbars attached. Will contain mean, standard error
        (1-sigma), number of samples, block size for independent samples. 
    """
    if verbose:
        print("# Reblock based on autocorrelation time")
    Nmax = int(numpy.log2(len(y)))
    Ndata = []
    tacs = []
    for i in range(Nmax):
        n = int(len(y) / 2**i)
        Ndata += [n]
        tacs += [autocorr_gw2010(y[:n])]
    if verbose:
        for n, tac in zip(reversed(Ndata), reversed(tacs)):
            print("nsamples, tac = {}, {}".format(n, tac))

    # block_size = int(numpy.round(numpy.max(tacs)))
    block_size = int(numpy.ceil(tacs[0])) # should take the one with the largest sample size
    nblocks = len(y) // block_size
    yblocked = []

    for i in range(nblocks):
        offset = i * block_size
        if i == nblocks-1: # including everything that's left
            yblocked += [numpy.mean(y[offset :])]
        else:
            yblocked += [numpy.mean(y[offset : offset + block_size])]

    yavg = numpy.mean(yblocked)
    ystd = numpy.std(yblocked) / numpy.sqrt(nblocks)

    df = pd.DataFrame(
        {
            "%s_ac" % name: [yavg],
            "%s_error_ac" % name: [ystd],
            "%s_nsamp_ac" % name: [nblocks],
            "ac": [block_size],
        }
    )

    return df
