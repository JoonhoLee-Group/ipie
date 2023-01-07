
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
#

"""Routines for performing propagation of a walker"""
import sys

from ipie.propagation.continuous import Continuous


# TODO: Fix for discrete transformation.
def get_propagator_driver(system, hamiltonian, trial, qmc, options={}, verbose=False):
    hs = options.get("hubbard_stratonovich", "continuous")
    assert not ("discrete" in hs)
    return Continuous(system, hamiltonian, trial, qmc, options=options, verbose=verbose)
