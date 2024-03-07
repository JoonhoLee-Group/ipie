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

import numpy


def jackknife_ratios(num: numpy.ndarray, denom: numpy.ndarray):
    r"""Jackknife estimation of standard deviation of the ratio of means.

    Parameters
    ----------
    num : :class:`np.ndarray
        Numerator samples.
    denom : :class:`np.ndarray`
        Denominator samples.

    Returns
    -------
    mean : :class:`np.ndarray`
        Ratio of means.
    sigma : :class:`np.ndarray`
        Standard deviation of the ratio of means.
    """
    n_samples = num.size
    num_mean = numpy.mean(num)
    denom_mean = numpy.mean(denom)
    mean = num_mean / denom_mean
    jackknife_estimates = numpy.zeros(n_samples, dtype=num.dtype)
    for i in range(n_samples):
        mean_num_i = (num_mean * n_samples - num[i]) / (n_samples - 1)
        mean_denom_i = (denom_mean * n_samples - denom[i]) / (n_samples - 1)
        jackknife_estimates[i] = (mean_num_i / mean_denom_i).real
    mean = numpy.mean(jackknife_estimates)
    sigma = numpy.sqrt((n_samples - 1) * numpy.var(jackknife_estimates))
    return mean, sigma
