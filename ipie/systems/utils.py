
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
#          Joonho Lee
#

import sys

import numpy

from ipie.systems.generic import Generic
from ipie.utils.mpi import get_shared_array, have_shared_mem


def get_system(sys_opts=None, verbose=0, comm=None):
    """Wrapper to select system class

    Parameters
    ----------
    sys_opts : dict
        System input options.
    verbose : bool
        Output verbosity.

    Returns
    -------
    system : object
        System class.
    """
    if not ("name" in sys_opts):
        sys_opts["name"] = "Generic"

    if sys_opts["name"] == "UEG":
        system = UEG(sys_opts, verbose)
    elif (
        sys_opts["name"] == "Hubbard"
        or sys_opts["name"] == "HubbardHolstein"
        or sys_opts["name"] == "Generic"
    ):
        nup, ndown = sys_opts.get("nup"), sys_opts.get("ndown")
        if nup is None or ndown is None:
            if comm.rank == 0:
                print("# Error: Number of electrons not specified.")
                sys.exit()
        nelec = (nup, ndown)
        system = Generic(nelec, sys_opts, verbose)
    else:
        if comm.rank == 0:
            print("# Error: unrecognized system name {}.".format(sys_opts["name"]))
            sys.exit()

    return system
