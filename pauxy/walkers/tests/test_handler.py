import pytest
import numpy

from mpi4py import MPI
comm = MPI.COMM_WORLD
numpy.random.seed(7)
skip = comm.size == 1

@pytest.mark.unit
@pytest.mark.skipif(skip, reason="Test should be run on multiple cores.")
def test_pair_branch():
    if comm.rank == 0:
        weights = [0.001, 1.0148, 4.348]
    else:
        weights = [1.2, 2.348, 4.4]
    walker_info = []
    for i, w in enumerate(weights):
        walker_info.append([w,1,comm.rank,comm.rank])
    comm.barrier()
    glob_inf = comm.allgather(walker_info)
    # print(comm.rank, glob_inf)
    # comm.barrier()
    min_weight = 0.1
    max_weight = 4.0
    # Unpack lists
    glob_inf = numpy.array([item for sub in glob_inf for item in sub])
    # glob_inf.sort(key=lambda x: x[0])
    sort = numpy.argsort(glob_inf[:,0], kind='mergesort')
    isort = numpy.argsort(sort, kind='mergesort')
    glob_inf = glob_inf[sort]
    s = 0
    e = len(glob_inf) - 1
    while s < e:
        if glob_inf[s][0] < min_weight or glob_inf[e][0] > max_weight:
            # sum of paired walker weights
            wab = glob_inf[s][0] + glob_inf[e][0]
            r = numpy.random.rand()
            if r < glob_inf[e][0] / wab:
                # clone large weight walker
                glob_inf[e][0] = 0.5 * wab
                glob_inf[e][1] = 2
                # Processor we will send duplicated walker to
                glob_inf[e][3] = glob_inf[s][2]
                # Kill small weight walker
                glob_inf[s][0] = 0.0
                glob_inf[s][1] = 0
                glob_inf[s][3] = glob_inf[e][2]
            else:
                # clone small weight walker
                glob_inf[s][0] = 0.5 * wab
                glob_inf[s][1] = 2
                # Processor we will send duplicated walker to
                glob_inf[s][3] = glob_inf[e][2]
                # Kill small weight walker
                glob_inf[e][0] = 0.0
                glob_inf[e][1] = 0
                glob_inf[e][3] = glob_inf[s][2]
            s += 1
            e -= 1
        else:
            break
    glob_inf = glob_inf[isort]
    reqs = []
    nw = len(weights)
    for walker in glob_inf[comm.rank*nw:(comm.rank+1)*nw]:
        if walker[1] > 1:
            tag = comm.rank*len(walker_info) + walker[3]
            reqs.append(comm.isend(comm.rank*numpy.ones(2),
                        dest=int(round(walker[3])), tag=tag))
    buff = []
    for walker in glob_inf[comm.rank*nw:(comm.rank+1)*nw]:
        if walker[1] == 0:
            tag = walker[3]*len(walker_info) + comm.rank
            buff.append(comm.recv(source=int(round(walker[3])), tag=tag))
    for r in reqs:
        r.wait()
    if comm.rank == 0:
        assert len(buff) == 2
        assert sum(buff[0]) == 2
        assert sum(buff[1]) == 0
