
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

import numpy

from ipie.utils.backend import arraylib as xp


def set_rng_seed(seed, comm):
    if seed is None:
        # only set "random" part of seed on parent processor so we can reproduce
        # results in when running in parallel.
        if comm.rank == 0:
            seed = numpy.array([numpy.random.randint(0, 1e8)], dtype="i4")
            # Can't directly json serialise numpy arrays
        else:
            seed = numpy.empty(1, dtype="i4")
        comm.Bcast(seed, root=0)
        seed = seed[0]
    seed = seed + comm.rank
    xp.random.seed(seed)
    if comm.rank == 0:
        print("# random seed is {}".format(seed))
    return seed


def gpu_synchronize(gpu):
    if gpu:
        import cupy
        cupy.cuda.stream.get_current_stream().synchronize()

    return
