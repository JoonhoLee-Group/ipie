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

from ipie.systems.generic import Generic


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
    assert sys_opts is not None
    sys_type = sys_opts.get("name")
    if sys_type is None or sys_type == "Generic":
        nup, ndown = sys_opts.get("nup"), sys_opts.get("ndown")
        if nup is None or ndown is None:
            if comm.rank == 0:
                print("# Error: Number of electrons not specified.")
                sys.exit()
        nelec = (nup, ndown)
        system = Generic(nelec, verbose)
    else:
        if comm.rank == 0:
            print(f"# Error: unrecognized system name {sys_type}.")
        raise ValueError

    return system
