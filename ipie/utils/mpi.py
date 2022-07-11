"""MPI Helper functions."""
import bigmpi4py as BM
import numpy
from mpi4py import MPI


class MPIHandler(object):
    def __init__(self, comm, options={}, verbose=False):

        self.comm = comm  # global communicator
        self.shared_comm = get_shared_comm(comm)  # global communicator
        self.size = comm.Get_size()
        self.rank = comm.Get_rank()
        self.nmembers = options.get("nmembers", 1)  # Each group has 1 member by default

        self.ngroups = self.size // self.nmembers

        if verbose:
            print(
                "# MPIHandler detected {} groups with {} members each".format(
                    self.ngroups, self.nmembers
                )
            )

        try:
            assert self.size == self.nmembers * self.ngroups
        except AssertionError:
            print("# MPI nmembers should divide the total MPI size")
            exit()

        self.color = self.rank // self.nmembers
        self.srank = self.rank - self.color * self.nmembers
        self.scomm = comm.Split(color=self.color, key=self.srank)
        self.ssize = self.scomm.Get_size()

        self.senders = numpy.array([i for i in range(self.ssize)])
        self.receivers = numpy.array([i for i in range(1, self.ssize)] + [0])

        # print("self.rank = {}".format(self.rank))
        # print("rank, color, key = {}, {}, {}, {}".format(self.rank, self.color, self.srank, self.scomm.Get_rank()))
        # print("self.srank = {}".format(self.srank))
        # print("self.ssize = {}".format(self.ssize))
        # print("self.ngroups = {}".format(self.ngroups))
        # print("# created {} of {} but originally {} of {}".format(self.srank, self.ssize, self.rank, self.size))
        assert self.ssize == self.nmembers
        assert self.srank == self.scomm.Get_rank()

    def scatter_group(self, array, root=0):  # scatter within a group
        return BM.scatter(array, self.scomm, root=0)

    def allreduce_group(self, array, root=0):  # scatter within a group
        return self.scomm.allreduce(array, op=MPI.SUM)

    def scatter(self, array, root=0):  # scatter globally
        return BM.scatter(array, self.comm, root=0)


def get_shared_comm(comm, verbose=False):
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
    try:
        win = MPI.Win.Allocate_shared(1, 1, comm=comm)
        return True
    except:
        return False
