class FakeComm:
    """Fake MPI communicator class to reduce logic."""

    def __init__(self):
        self.rank = 0
        self.size = 1

    def Barrier(self):
        pass

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Gather(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def Bcast(self, sendbuf, root=0):
        return sendbuf

    def bcast(self, sendbuf, root=0):
        return sendbuf

    def isend(self, sendbuf, dest=None, tag=None):
        return FakeReq()

    def recv(self, source=None, root=0):
        pass

    def Reduce(self, sendbuf, recvbuf, op=None):
        recvbuf[:] = sendbuf


class FakeReq:
    def __init__(self):
        pass

    def wait(self):
        pass
