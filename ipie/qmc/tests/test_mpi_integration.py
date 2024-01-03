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
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee <linusjoonho@gmail.com>
#

import json
import os
import sys
import tempfile
import uuid

import numpy as np

from ipie.analysis.extraction import extract_test_data_hdf5
from ipie.config import MPI
from ipie.qmc.calc import get_driver, read_input

try:
    from ipie.legacy.qmc.calc import get_driver as get_legacy_driver

    _built_legacy = True
except:
    _built_legacy = False


comm = MPI.COMM_WORLD
serial_test = comm.size == 1
# Unique filename to avoid name collision when running through CI.
if comm.rank == 0:
    test_id = str(uuid.uuid1())
else:
    test_id = None
test_id = comm.bcast(test_id, root=0)

_data_dir = os.path.abspath(os.path.dirname(__file__)) + "/reference_data/"
# glob is a bit dangerous.
# _test_dirs = [d for d in glob.glob(_data_dir+'/*') if os.path.isdir(d)]
_test_dirs = [
    "h10_cc-pvtz_batched",
    "h10_cc-pvtz_pair_branch",
    "benzene_cc-pvdz_batched",  # disabling
    "benzene_cc-pvdz_chunked",
]
_tests = [(_data_dir + d + "/input.json", _data_dir + d + "/reference.json") for d in _test_dirs]
_legacy_test_dirs = [
    "4x4_hubbard_discrete",
    "neon_cc-pvdz_rhf",
    "ft_4x4_hubbard_discrete",
    "ft_ueg_ecut1.0_rs1.0",
    "ueg_ecut2.5_rs2.0_ne14",
]
_legacy_tests = [
    (_data_dir + d + "/input.json", _data_dir + d + "/reference.json") for d in _legacy_test_dirs
]


def compare_test_data(ref, test):
    comparison = {}
    for k, v in ref.items():
        if k == "sys_info":
            continue
        try:
            comparison[k] = (
                np.array(ref[k]),
                np.array(test[k]),
                np.max(np.abs(np.array(ref[k]) - np.array(test[k]))) < 1e-10,
            )
        except KeyError:
            print(f"# Issue with test data key {k}")
    return comparison


def run_test_system(input_file, benchmark_file, legacy_job=False):
    comm = MPI.COMM_WORLD
    input_dict = read_input(input_file, comm)
    if input_dict["system"].get("integrals") is not None:
        input_dict["system"]["integrals"] = input_file[:-10] + "afqmc.h5"
        input_dict["trial"]["filename"] = input_file[:-10] + "afqmc.h5"
    elif ("hamiltonian" in input_dict) and (input_dict["hamiltonian"].get("integrals") is not None):
        input_dict["hamiltonian"]["integrals"] = input_file[:-10] + "afqmc.h5"
        input_dict["trial"]["filename"] = input_file[:-10] + "afqmc.h5"
    with tempfile.NamedTemporaryFile() as tmpf:
        input_dict["estimators"]["filename"] = tmpf.name
        if _built_legacy and legacy_job:
            from ipie.legacy.qmc.calc import get_driver as get_legacy_driver

            input_dict["qmc"]["batched"] = False
            afqmc = get_legacy_driver(input_dict, comm)
            afqmc.run(comm=comm)
            afqmc.finalise(comm)
        else:
            afqmc = get_driver(input_dict, comm)
            afqmc.run(estimator_filename=tmpf.name)
            afqmc.finalise()
        if comm.rank == 0:
            with open(benchmark_file, "r") as f:
                ref_data = json.load(f)
            skip_val = ref_data.get("extract_skip_value", 10)
            test_data = extract_test_data_hdf5(tmpf.name, skip=skip_val)
            comparison = compare_test_data(ref_data, test_data)
        else:
            comparison = None
        comm.barrier()
        comparison = comm.bcast(comparison)
    return comparison


if __name__ == "__main__":
    err_count = 0
    err_msg = []
    for test_name, (ind, outd) in zip(_test_dirs, _tests):
        comparison = run_test_system(ind, outd)
        local_err_count = 0
        if comm.rank == 0:
            for k, v in comparison.items():
                if not v[-1]:
                    local_err_count += 1
                    print(
                        f" *** FAILED **** : mismatch between benchmark and test run: {test_name}"
                    )
                    err_msg.append(
                        f" *** FAILED **** : mismatch between benchmark and test run: {test_name}"
                    )
                    err_count += 1
                    print(f"name = {k}\n ref = {v[0]}\n test = {v[1]}\n delta = {v[0]-v[1]}\n")
        else:
            err_count = None
        if local_err_count == 0 and comm.rank == 0:
            print(f"*** PASSED : {test_name} ***")
    err_count = comm.bcast(err_count)
    err_msg = comm.bcast(err_msg)
    for test_name, (ind, outd) in zip(_legacy_test_dirs, _legacy_tests):
        comparison = run_test_system(ind, outd, legacy_job=True)
        local_err_count = 0
        if comm.rank == 0:
            for k, v in comparison.items():
                if not v[-1]:
                    local_err_count += 1
                    print(f" *** FAILED *** : mismatch between benchmark and test run: {test_name}")
                    err_msg.append(
                        f" *** FAILED **** : mismatch between benchmark and test run: {test_name}"
                    )
                    err_count += 1
                    print(f"name = {k}\n ref = {v[0]}\n test = {v[1]}\n delta = {v[0]-v[1]}\n")
        else:
            err_count = None
        if local_err_count == 0 and comm.rank == 0:
            print(f"*** PASSED : {test_name} ***")
    err_count = comm.bcast(err_count)
    err_msg = comm.bcast(err_msg)
    if local_err_count > 0 and comm.rank == 0:
        for msg in err_msg:
            print(msg)
    sys.exit(err_count)
