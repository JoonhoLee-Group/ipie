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

"""MPI Helper functions."""
import numpy


def make_splits_displacements(ntotal, nsplit):
    nt = int(ntotal // nsplit)
    split_sizes_t = numpy.array([nt for i in range(nsplit)])
    residual = ntotal - nt * nsplit
    for i in range(residual):
        split_sizes_t[nsplit - 1 - i] += 1
    displacements_t = numpy.insert(numpy.cumsum(split_sizes_t), 0, 0)[0:-1]
    assert numpy.sum(split_sizes_t) == ntotal
    return split_sizes_t, displacements_t


class MPIHandler(object):
    def __init__(self, comm, nmembers: int = 1, verbose: bool = False):
        self.comm = comm  # global communicator
        self.shared_comm = get_shared_comm(comm)  # global communicator
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()

        self.nmembers = nmembers
        self.ngroups = self.size // self.nmembers

        if verbose:
            print(f"# MPIHandler detected {self.ngroups} groups with {self.nmembers} members each")

        try:
            assert self.size == self.nmembers * self.ngroups
        except AssertionError:
            raise ValueError("# MPI nmembers should divide the total MPI size")

        self.color = self.rank // self.nmembers
        self.srank = self.rank - self.color * self.nmembers
        self.scomm = comm.Split(color=self.color, key=self.srank)
        self.ssize = self.scomm.Get_size()

        self.senders = numpy.array([i for i in range(self.ssize)])
        self.receivers = numpy.array([i for i in range(1, self.ssize)] + [0])

        assert self.ssize == self.nmembers
        assert self.srank == self.scomm.Get_rank()

    def scatter_group(self, array, root=0):  # scatter within a group
        ntotal = len(array)
        nsplit = self.ssize
        split_sizes, displacements = make_splits_displacements(ntotal, nsplit)
        from mpi4py import MPI

        if isinstance(array, list):
            if isinstance(array[0], int):
                my_array = numpy.zeros(split_sizes[self.srank], dtype=numpy.int64)
                tmp = numpy.array(array)
                self.scomm.Scatterv([tmp, split_sizes, displacements, MPI.INT64_T], my_array, root)
        elif isinstance(array, numpy.ndarray):
            if len(array.shape) == 2:
                ncols = array.shape[1]
                my_array = numpy.zeros((split_sizes[self.srank], ncols), dtype=array.dtype)
                self.scomm.Scatterv(
                    [array, split_sizes * ncols, displacements * ncols, MPI.DOUBLE],
                    my_array,
                    root,
                )
        else:
            print("scatter_group not yet implemented for this array type")
            exit()
        return my_array

    def allreduce_group(self, array, root=0):  # allreduce within a group
        from mpi4py import MPI

        return self.scomm.allreduce(array, op=MPI.SUM)


def get_shared_comm(comm, verbose=False):
    from mpi4py import MPI

    try:
        return comm.Split_type(MPI.COMM_TYPE_SHARED)
    except:
        if verbose:
            print("# No MPI shared memory available.")
        return None


def get_shared_array(comm, shape, dtype, verbose=False):
    """Get shared memory numpy array.

    Parameters
    ----------
    comm : `mpi4py.MPI`
    """
    size = numpy.prod(shape)
    from mpi4py import MPI

    try:
        itemsize = numpy.dtype(dtype).itemsize
        if comm.rank == 0:
            nbytes = size * itemsize
        else:
            nbytes = 0
        win = MPI.Win.Allocate_shared(nbytes, itemsize, comm=comm)
        buf, itemsize = win.Shared_query(0)
        assert itemsize == numpy.dtype(dtype).itemsize
        buf = numpy.array(buf, dtype="B", copy=False)
        return numpy.ndarray(buffer=buf, dtype=dtype, shape=shape)
    except AttributeError:
        if verbose:
            print("# No MPI shared memory available.", comm.rank)
        return numpy.zeros(shape, dtype=dtype)


def have_shared_mem(comm):
    from mpi4py import MPI

    try:
        MPI.Win.Allocate_shared(1, 1, comm=comm)
        return True
    except:
        return False
