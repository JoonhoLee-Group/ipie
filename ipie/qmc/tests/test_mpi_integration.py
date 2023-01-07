
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
import uuid

import numpy as np
import pytest
from mpi4py import MPI

from ipie.analysis.extraction import extract_test_data_hdf5
from ipie.qmc.calc import get_driver, read_input

try:
    import pytest_mpi

    have_pytest_mpi = True
except ImportError:
    have_pytest_mpi = False


comm = MPI.COMM_WORLD
serial_test = comm.size == 1
# Unique filename to avoid name collision when running through CI.
test_id = str(uuid.uuid1())
output_file = "estimates_{}.h5".format(test_id)

_data_dir = os.path.abspath(os.path.dirname(__file__)) + "/reference_data/"
# glob is a bit dangerous.
# _test_dirs = [d for d in glob.glob(_data_dir+'/*') if os.path.isdir(d)]
_test_dirs = [
    "4x4_hubbard_discrete",
    "ft_4x4_hubbard_discrete",
    "ft_ueg_ecut1.0_rs1.0",
    "ueg_ecut2.5_rs2.0_ne14",
    "h10_cc-pvtz_batched",
    "h10_cc-pvtz_pair_branch",
    "neon_cc-pvdz_rhf",
    "benzene_cc-pvdz_batched",  # disabling
    "benzene_cc-pvdz_chunked",
]
_tests = [
    (_data_dir + d + "/input.json", _data_dir + d + "/reference.json")
    for d in _test_dirs
]


def compare_test_data(ref, test):
    for k, v in ref.items():
        if k == "sys_info":
            continue
        try:
            print(k, np.array(ref[k]), np.array(test[k]))
            return np.max(np.abs(np.array(ref[k]) - np.array(test[k]))) < 1e-10
        except ValueError:
            raise RuntimeError("Issue with test vs reference data")


def run_test_system(input_file, benchmark_file):
    comm = MPI.COMM_WORLD
    input_dict = read_input(input_file, comm)
    if input_dict["system"].get("integrals") is not None:
        input_dict["system"]["integrals"] = input_file[:-10] + "afqmc.h5"
        input_dict["trial"]["filename"] = input_file[:-10] + "afqmc.h5"
    elif ("hamiltonian" in input_dict) and (
        input_dict["hamiltonian"].get("integrals") is not None
    ):
        input_dict["hamiltonian"]["integrals"] = input_file[:-10] + "afqmc.h5"
        input_dict["trial"]["filename"] = input_file[:-10] + "afqmc.h5"
    input_dict["estimators"]["filename"] = output_file
    afqmc = get_driver(input_dict, comm)
    afqmc.run(comm=comm)
    if comm.rank == 0:
        with open(benchmark_file, "r") as f:
            ref_data = json.load(f)
        skip_val = ref_data.get("extract_skip_value", 10)
        test_data = extract_test_data_hdf5(output_file, skip=skip_val)
        try:
            _passed = compare_test_data(ref_data, test_data)
        except RuntimeError:
            _passed = False
    else:
        _passed = None
    passed = comm.bcast(_passed)
    assert passed


@pytest.mark.mpi
@pytest.mark.skipif(serial_test, reason="Test should be run on multiple cores.")
@pytest.mark.skipif(not have_pytest_mpi, reason="Test requires pytest-mpi plugin.")
@pytest.mark.parametrize("input_dir, benchmark_dir", _tests)
def test_system_mpi(input_dir, benchmark_dir):
    run_test_system(input_dir, benchmark_dir)


def teardown_module():
    cwd = os.getcwd()
    files = [output_file]
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass
