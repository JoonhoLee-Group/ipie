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


import h5py
import numpy
import pandas as pd


def extract_hdf5_data(filename, block_idx=1):
    shapes = {}
    with h5py.File(filename, "r") as fh5:
        keys = fh5[f"block_size_{block_idx}/data/"].keys()
        shape_keys = fh5[f"block_size_{block_idx}/shape/"].keys()
        data = numpy.concatenate([fh5[f"block_size_{block_idx}/data/{d}"][:] for d in keys])
        for k in shape_keys:
            shapes[k] = {
                "names": fh5[f"block_size_{block_idx}/names/{k}"][()],
                "shape": fh5[f"block_size_{block_idx}/shape/{k}"][:],
                "offset": fh5[f"block_size_{block_idx}/offset/{k}"][()],
                "size": fh5[f"block_size_{block_idx}/size/{k}"][()],
                "scalar": bool(fh5[f"block_size_{block_idx}/scalar/{k}"][()]),
                "num_walker_props": fh5[f"block_size_{block_idx}/num_walker_props"][()],
                "walker_header": fh5[f"block_size_{block_idx}/walker_prop_header"][()],
            }
        size_keys = fh5[f"block_size_{block_idx}/max_block"].keys()
        max_block = sum(fh5[f"block_size_{block_idx}/max_block/{d}"][()] for d in size_keys)

    return data[: max_block + 1], shapes


def extract_observable(filename, name="energy", block_idx=1):
    data, info = extract_hdf5_data(filename, block_idx=block_idx)
    obs_info = info.get(name)
    if obs_info is None:
        raise RuntimeError(f"Unknown value for name={name}")
    obs_slice = slice(obs_info["offset"], obs_info["offset"] + obs_info["size"])
    if obs_info["scalar"]:
        obs_data = data[:, obs_slice].reshape((-1,) + tuple(obs_info["shape"]))
        nwalk_prop = obs_info["num_walker_props"]
        weight_data = data[:, :nwalk_prop].reshape((-1, nwalk_prop))
        results = pd.DataFrame(numpy.hstack([weight_data, obs_data]))
        header = list(obs_info["walker_header"]) + obs_info["names"].split()
        results.columns = [n.decode("utf-8") for n in header]
        return results
    else:
        obs_data = data[:, obs_slice]
        nsamp = data.shape[0]
        walker_averaged = obs_data[:, :-1] / obs_data[:, -1].reshape((nsamp, -1))
        return walker_averaged.reshape((nsamp,) + tuple(obs_info["shape"]))
