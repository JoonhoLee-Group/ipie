"""MPI Helper functions."""
import numpy
from mpi4py import MPI

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
        buf = numpy.array(buf, dtype='B', copy=False)
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
