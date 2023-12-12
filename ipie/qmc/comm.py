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

from dataclasses import dataclass
from typing import TypeVar


class FakeComm:
    """Fake MPI communicator class to reduce logic."""

    def __init__(self):
        self.rank = 0
        self.size = 1
        self.buffer = {}

    def Barrier(self):
        pass

    def barrier(self):
        pass

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def Gather(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def gather(self, sendbuf, root=0):
        return [sendbuf]

    def Allgather(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def Bcast(self, sendbuf, root=0):
        return sendbuf

    def bcast(self, sendbuf, root=0):
        return sendbuf

    def isend(self, sendbuf, dest=None, tag=None):
        return FakeReq()

    def Isend(self, sendbuf, dest=None, tag=None):
        self.buffer[tag] = sendbuf
        return FakeReq()

    def recv(self, source=None, root=0):
        pass

    def Recv(self, recvbuff, source=None, root=0, tag=0):
        if self.buffer.get(tag) is not None:
            recvbuff[:] = self.buffer[tag].copy()

    def Allreduce(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def allreduce(self, sendbuf, op=None, root=0):
        return sendbuf.copy()

    def Split(self, color: int = 0, key: int = 0):
        return self

    def Reduce(self, sendbuf, recvbuf, op=None, root=0):
        recvbuf[:] = sendbuf

    def Scatter(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def Scatterv(self, sendbuf, recvbuf, root=0):
        recvbuf[:] = sendbuf

    def scatter(self, sendbuf, root=0):
        assert sendbuf.shape[0] == 1, "Incorrect array shape in FakeComm.scatter"
        return sendbuf[0]


class FakeReq:
    def __init__(self):
        pass

    def wait(self):
        pass


@dataclass
class MPI:
    COMM_WORLD = FakeComm()
    SUM = None
    COMM_SPLIT_TYPE_SHARED = None
    COMM_TYPE_SHARED = None
    DOUBLE = None
    INT64_T = None
    Win = None
    IntraComm = TypeVar("IntraComm")
