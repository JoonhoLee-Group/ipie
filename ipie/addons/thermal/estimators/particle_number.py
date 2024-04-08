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

from ipie.utils.backend import arraylib as xp
from ipie.estimators.estimator_base import EstimatorBase
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G


def particle_number(dmat):
    """Compute average particle number from the thermal 1RDM.

    Parameters
    ----------
    dmat : :class:`numpy.ndarray`
        Thermal 1RDM.

    Returns
    -------
    nav : float
        Average particle number.
    """
    nav = dmat[0].trace() + dmat[1].trace()
    return nav


class ThermalNumberEstimator(EstimatorBase):
    def __init__(self, hamiltonian=None, trial=None, filename=None):
        # We define a dictionary to contain whatever we want to compute.
        # Note we typically want to separate the numerator and denominator of
        # the estimator.
        # We require complex valued buffers for accumulation
        self._data = {
            "NavNumer": 0.0j,
            "NavDenom": 0.0j,
            "Nav": 0.0j,
        }

        # We also need to specify the shape of the desired estimator
        self._shape = (len(self.names),)

        # Optional but good to know (we can redirect to custom filepath (ascii)
        # and / or print to stdout but we shouldnt do this for non scalar
        # quantities.
        self._data_index = {k: i for i, k in enumerate(list(self._data.keys()))}
        self.print_to_stdout = True
        self.ascii_filename = filename

        # Must specify that we're dealing with array valued estimator.
        self.scalar_estimator = True

    def compute_estimator(self, walkers, hamiltonian, trial):
        for iw in range(walkers.nwalkers):
            # Want the full Green's function when calculating observables.
            walkers.calc_greens_function(iw, slice_ix=walkers.stack[iw].nslice)
            nav_iw = particle_number(one_rdm_from_G(
                        xp.array([walkers.Ga[iw], walkers.Gb[iw]])))
            self._data["NavNumer"] += walkers.weight[iw] * nav_iw.real

        self._data["NavDenom"] = sum(walkers.weight)

    def get_index(self, name):
        index = self._data_index.get(name, None)

        if index is None:
            raise RuntimeError(f"Unknown estimator {name}")

        return index

    def post_reduce_hook(self, data):
        ix_proj = self._data_index["Nav"]
        ix_nume = self._data_index["NavNumer"]
        ix_deno = self._data_index["NavDenom"]
        data[ix_proj] = data[ix_nume] / data[ix_deno]
