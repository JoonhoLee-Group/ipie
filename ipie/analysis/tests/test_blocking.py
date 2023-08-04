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
#          Joonho Lee <linusjoonho@gmail.com>
#

import json
import os
import tempfile

import h5py
import numpy as np
import pytest

from ipie.analysis.blocking import analyse_estimates, reblock_minimal
from ipie.estimators.utils import H5EstimatorHelper


@pytest.mark.unit
def test_analyse_estimates():
    np.random.seed(7)
    nestim = 11
    nblock = 500
    data = np.random.random((nblock, nestim)) + 1j * np.random.random((nblock, nestim))
    header = [
        "Iteration",
        "WeightFactor",
        "Weight",
        "ENumer",
        "EDenom",
        "ETotal",
        "E1Body",
        "E2Body",
        "EHybrid",
        "Overlap",
        "Time",
    ]
    # fake some data
    metadata = {
        "system": {"nbasis": 1, "name": "Generic", "nup": 1, "ndown": 1, "integral_file": "none"},
        "qmc": {"nsteps": 1, "dt": 0.05},
        "propagators": {"free_projection": False},
        "trial": {"energy": -1.0},
    }
    with tempfile.NamedTemporaryFile() as tmpf:
        with h5py.File(tmpf.name, "w") as fh5:
            fh5["basic/headers"] = np.array(header).astype("S")
            fh5["metadata"] = json.dumps(metadata)
        helper = H5EstimatorHelper(tmpf.name, "basic")
        for i in range(nblock):
            helper.push(data[i], "energies")
            helper.increment()

        data = analyse_estimates([tmpf.name], start_time=0)
        assert np.allclose(data["ETotal"].values, [4.8279208658716e-01])
        assert np.allclose(
            data["ETotal_ac"].values,
            # [4.832190780587988e-01] # old reblock results
            [4.830773191766473e-01],
        )
        assert np.allclose(data["ETotal_error"].values[0], [1.1124618641309e-02])
        assert np.allclose(
            data["ETotal_error_ac"].values,
            # [1.301950758507343e-02] # old reblock results
            [1.257340745340220e-02],
        )


@pytest.mark.unit
def test_analyse_estimates_textfile():
    np.random.seed(7)
    nestim = 11
    nblock = 500
    data = np.random.random((nblock, nestim)) + 1j * np.random.random((nblock, nestim))
    header = [
        "Iteration",
        "WeightFactor",
        "Weight",
        "ENumer",
        "EDenom",
        "ETotal",
        "E1Body",
        "E2Body",
        "EHybrid",
        "Overlap",
        "Time",
    ]
    # fake some data
    with tempfile.NamedTemporaryFile() as tmpf:
        with open(tmpf.name, "w") as f:
            f.write(" ".join(h for h in header) + "\n")
            for i in range(nblock):
                f.write(" ".join(f"{val.real:13.10e}" for val in data[i]) + "\n")
            f.write("# End Time\n")

        data = reblock_minimal([tmpf.name], start_block=1)

        assert np.allclose(
            data["ETotal_ac"].values,
            # [4.832190780587988e-01]
            [4.830773191766062e-01],
        )
        assert np.allclose(
            data["ETotal_error_ac"].values,
            # [1.301950758507343e-02]
            [1.257340745340048e-02],
        )
