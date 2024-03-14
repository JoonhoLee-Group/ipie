import time

import numpy

from ipie.config import MPI
from ipie.utils.backend import arraylib as xp


class PopControllerTimer:
    def __init__(self):
        self.start_time_const = 0.0
        self.communication_time = 0.0
        self.non_communication_time = 0.0
        self.recv_time = 0.0
        self.send_time = 0.0

    def start_time(self):
        self.start_time_const = time.time()

    def add_non_communication(self):
        self.non_communication_time += time.time() - self.start_time_const

    def add_communication(self):
        self.communication_time += time.time() - self.start_time_const

    def add_recv_time(self):
        self.recv_time += time.time() - self.start_time_const

    def add_send_time(self):
        self.send_time += time.time() - self.start_time_const


class PopController:
    def __init__(
        self,
        num_walkers_local,
        num_steps,
        mpi_handler=None,
        pop_control_method="pair_branch",
        min_weight=0.1,
        max_weight=4,
        reconfiguration_freq=50,
        verbose=False,
    ):
        self.verbose = verbose

        self.num_walkers_local = num_walkers_local
        self.num_steps = num_steps
        self.mpi_handler = mpi_handler

        self.method = pop_control_method
        if verbose:
            print(f"# Using {self.method} population control " "algorithm.")

        self.min_weight = min_weight
        self.max_weight = max_weight
        self.reconfiguration_counter = 0
        self.reconfiguration_freq = reconfiguration_freq

        self.mpi_handler = mpi_handler

        if self.mpi_handler is not None:
            self.size = self.mpi_handler.size
            self.ntot_walkers = num_walkers_local * self.size
        else:
            self.size = 1
            self.ntot_walkers = num_walkers_local

        self.target_weight = self.ntot_walkers
        self.total_weight = self.ntot_walkers

        if verbose:
            print(f"# target weight is {self.target_weight}")
            print(f"# total weight is {self.total_weight}")

        self.timer = PopControllerTimer()

    def pop_control(self, walkers, comm):
        self.timer.start_time()
        if self.ntot_walkers == 1:
            return
        weights = numpy.abs(xp.array(walkers.weight))
        global_weights = numpy.empty(len(weights) * comm.size)
        self.timer.add_non_communication()
        self.timer.start_time()
        if self.method == "comb":
            comm.Allgather(weights, global_weights)
            total_weight = sum(global_weights)
        else:
            sum_weights = numpy.sum(weights)
            total_weight = numpy.empty(1, dtype=numpy.float64)
            if hasattr(sum_weights, "get"):
                sum_weights = sum_weights.get()
            comm.Reduce(sum_weights, total_weight, op=MPI.SUM, root=0)
            comm.Bcast(total_weight, root=0)
            total_weight = total_weight[0]

        self.timer.add_communication()
        self.timer.start_time()

        # Rescale weights to combat exponential decay/growth.
        scale = total_weight / self.target_weight
        if total_weight < 1e-8:
            if comm.rank == 0:
                print(f"# Warning: Total weight is {total_weight:13.8e}")
                print("# Something is seriously wrong.")
            raise ValueError
        self.total_weight = total_weight
        # Todo: Just standardise information we want to send between routines.
        walkers.unscaled_weight = walkers.weight
        walkers.weight = walkers.weight / scale
        if self.method == "comb":
            global_weights = global_weights / scale
            self.timer.add_non_communication()
            comb(walkers, comm, global_weights, self.target_weight, self.timer)
        elif self.method == "pair_branch":
            pair_branch(walkers, comm, self.max_weight, self.min_weight, self.timer)
        elif self.method == "stochastic_reconfiguration":
            self.reconfiguration_counter += 1
            if self.reconfiguration_counter % self.reconfiguration_freq == 0:
                stochastic_reconfiguration(walkers, comm, self.timer)
                self.reconfiguration_counter = 0
        else:
            if comm.rank == 0:
                print("Unknown population control method.")


def get_buffer(walkers, iw):
    """Get iw-th walker buffer for MPI communication
    iw : int
        the walker index of interest
    Returns
    -------
    buff : dict
        Relevant walker information for population control.
    """
    s = 0
    buff = xp.zeros(walkers.buff_size, dtype=numpy.complex128)
    for d in walkers.buff_names:
        data = walkers.__dict__[d]
        if data is None:
            continue
        assert data.size % walkers.nwalkers == 0  # Only walker-specific data is being communicated
        if isinstance(data[iw], (xp.ndarray)):
            buff[s : s + data[iw].size] = xp.array(data[iw].ravel())
            s += data[iw].size
        elif isinstance(data[iw], list):  # when data is list
            for l in data[iw]:
                if isinstance(l, (xp.ndarray)):
                    buff[s : s + l.size] = xp.array(l.ravel())
                    s += l.size
                elif isinstance(l, (int, float, complex, numpy.float64, numpy.complex128)):
                    buff[s : s + 1] = l
                    s += 1
        else:
            buff[s : s + 1] = xp.array(data[iw])
            s += 1
    return buff


def set_buffer(walkers, iw, buff):
    """Set walker buffer following MPI communication
    Parameters
    -------
    buff : dict
        Relevant walker information for population control.
    """
    s = 0
    for d in walkers.buff_names:
        data = walkers.__dict__[d]
        if data is None:
            continue
        assert data.size % walkers.nwalkers == 0  # Only walker-specific data is being communicated
        if isinstance(data[iw], xp.ndarray):
            walkers.__dict__[d][iw] = xp.array(
                buff[s : s + data[iw].size].reshape(data[iw].shape).copy()
            )
            s += data[iw].size
        elif isinstance(data[iw], list):
            for ix, l in enumerate(data[iw]):
                if isinstance(l, (xp.ndarray)):
                    walkers.__dict__[d][iw][ix] = xp.array(
                        buff[s : s + l.size].reshape(l.shape).copy()
                    )
                    s += l.size
                elif isinstance(l, (int, float, complex)):
                    walkers.__dict__[d][iw][ix] = buff[s]
                    s += 1
        else:
            if isinstance(walkers.__dict__[d][iw], (int, numpy.int64)):
                walkers.__dict__[d][iw] = int(buff[s].real)
            elif isinstance(walkers.__dict__[d][iw], (float, numpy.float64)):
                walkers.__dict__[d][iw] = buff[s].real
            else:
                walkers.__dict__[d][iw] = buff[s]
            s += 1


def comb(walkers, comm, weights, target_weight, timer=PopControllerTimer()):
    """Apply the comb method of population control / branching.

    See Booth & Gubernatis PRE 80, 046704 (2009).

    Parameters
    ----------
    comm : MPI communicator
    """
    # Need make a copy to since the elements in psi are only references to
    # walker objects in memory. We don't want future changes in a given
    # element of psi having unintended consequences.
    # todo : add phase to walker for free projection
    timer.start_time()
    if comm.rank == 0:
        parent_ix = numpy.zeros(len(weights), dtype="i")
    else:
        parent_ix = numpy.empty(len(weights), dtype="i")
    if comm.rank == 0:
        total_weight = sum(weights)
        cprobs = numpy.cumsum(weights)
        r = numpy.random.random()
        comb = [(i + r) * (total_weight / target_weight) for i in range(target_weight)]
        iw = 0
        ic = 0
        while ic < len(comb):
            if comb[ic] < cprobs[iw]:
                parent_ix[iw] += 1
                ic += 1
            else:
                iw += 1
        data = {"ix": parent_ix}
    else:
        data = None

    timer.add_non_communication()

    timer.start_time()
    data = comm.bcast(data, root=0)
    timer.add_communication()
    timer.start_time()
    parent_ix = data["ix"]
    # Keep total weight saved for capping purposes.
    # where returns a tuple (array,), selecting first element.
    kill = numpy.where(parent_ix == 0)[0]
    clone = numpy.where(parent_ix > 1)[0]
    reqs = []
    # First initiate non-blocking sends of walkers.
    timer.add_non_communication()
    timer.start_time()
    comm.barrier()
    timer.add_communication()
    for i, (c, k) in enumerate(zip(clone, kill)):
        # Sending from current processor?
        if c // walkers.nwalkers == comm.rank:
            timer.start_time()
            # Location of walker to clone in local list.
            clone_pos = c % walkers.nwalkers
            # copying walker data to intermediate buffer to avoid issues
            # with accessing walker data during send. Might not be
            # necessary.
            dest_proc = k // walkers.nwalkers
            buff = get_buffer(walkers, clone_pos)
            timer.add_non_communication()
            timer.start_time()
            reqs.append(comm.Isend(buff, dest=dest_proc, tag=i))
            timer.add_send_time()
    # Now receive walkers on processors where walkers are to be killed.
    for i, (c, k) in enumerate(zip(clone, kill)):
        # Receiving to current processor?
        if k // walkers.nwalkers == comm.rank:
            timer.start_time()
            # Processor we are receiving from.
            source_proc = c // walkers.nwalkers
            # Location of walker to kill in local list of walkers.
            kill_pos = k % walkers.nwalkers
            timer.add_non_communication()
            timer.start_time()
            comm.Recv(walkers.walker_buffer, source=source_proc, tag=i)
            # with h5py.File('walkers_recv.h5', 'w') as fh5:
            # fh5['walk_{}'.format(k)] = walkers.walker_buffer.copy()
            timer.add_recv_time()
            timer.start_time()
            set_buffer(walkers, kill_pos, walkers.walker_buffer)
            timer.add_non_communication()
            # with h5py.File('after_{}.h5'.format(comm.rank), 'a') as fh5:
            # fh5['walker_{}_{}_{}'.format(c,k,comm.rank)] = walkers.walkers[kill_pos].get_buffer()
    timer.start_time()
    # Complete non-blocking send.
    for rs in reqs:
        rs.wait()
    # Necessary?
    # if len(kill) > 0 or len(clone) > 0:
    # sys.exit()
    comm.Barrier()
    timer.add_communication()
    # Reset walker weight.
    # TODO: check this.
    # for w in walkers.walkers:
    # w.weight = 1.0
    timer.start_time()
    walkers.weight.fill(1.0)
    timer.add_non_communication()


def pair_branch(walkers, comm, max_weight, min_weight, timer=PopControllerTimer()):
    timer.start_time()
    walker_info_0 = xp.array(xp.abs(walkers.weight))
    timer.add_non_communication()

    timer.start_time()
    glob_inf = None
    glob_inf_0 = None
    glob_inf_1 = None
    glob_inf_2 = None
    glob_inf_3 = None
    if comm.rank == 0:
        glob_inf_0 = numpy.empty([comm.size, walkers.nwalkers], dtype=numpy.float64)
        glob_inf_1 = numpy.empty([comm.size, walkers.nwalkers], dtype=numpy.int64)
        glob_inf_1.fill(1)
        glob_inf_2 = numpy.array(
            [[r for i in range(walkers.nwalkers)] for r in range(comm.size)],
            dtype=numpy.int64,
        )
        glob_inf_3 = numpy.array(
            [[r for i in range(walkers.nwalkers)] for r in range(comm.size)],
            dtype=numpy.int64,
        )

    timer.add_non_communication()

    timer.start_time()
    if hasattr(walker_info_0, "get"):
        walker_info_0 = walker_info_0.get()
    comm.Gather(
        walker_info_0, glob_inf_0, root=0
    )  # gather |w_i| from all processors (comm.size x nwalkers)
    timer.add_communication()

    # Want same random number seed used on all processors
    timer.start_time()
    if comm.rank == 0:
        # Rescale weights.
        glob_inf = numpy.zeros((walkers.nwalkers * comm.size, 4), dtype=numpy.float64)
        glob_inf[:, 0] = glob_inf_0.ravel()  # contains walker |w_i|
        glob_inf[:, 1] = (
            glob_inf_1.ravel()
        )  # all initialized to 1 when it becomes 2 then it will be "branched"
        glob_inf[:, 2] = (
            glob_inf_2.ravel()
        )  # contain processor+walker indices (initial) (i.e., where walkers live)
        glob_inf[:, 3] = (
            glob_inf_3.ravel()
        )  # contain processor+walker indices (final) (i.e., where walkers live)
        sort = numpy.argsort(glob_inf[:, 0], kind="mergesort")
        isort = numpy.argsort(sort, kind="mergesort")
        glob_inf = glob_inf[sort]
        s = 0
        e = len(glob_inf) - 1
        tags = []
        # go through walkers pair-wise
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
                    send = glob_inf[s][2]
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
                    send = glob_inf[e][2]
                    # Kill small weight walker
                    glob_inf[e][0] = 0.0
                    glob_inf[e][1] = 0
                    glob_inf[e][3] = glob_inf[s][2]
                tags.append([send])
                s += 1
                e -= 1
            else:
                break
        nw = walkers.nwalkers
        glob_inf = glob_inf[isort].reshape((comm.size, nw, 4))
    else:
        data = None
        glob_inf = None
    timer.add_non_communication()
    timer.start_time()

    data = numpy.empty([walkers.nwalkers, 4], dtype=numpy.float64)
    # 0 = weight, 1 = status (live, branched, die), 2 = initial index, 3 = final index
    comm.Scatter(glob_inf, data, root=0)

    timer.add_communication()
    # Keep total weight saved for capping purposes.
    reqs = []
    for iw, walker in enumerate(data):
        if walker[1] > 1:
            timer.start_time()
            tag = comm.rank * walkers.nwalkers + walker[3]
            walkers.weight[iw] = walker[0]
            buff = get_buffer(walkers, iw)
            timer.add_non_communication()
            timer.start_time()
            reqs.append(comm.Isend(buff, dest=int(round(walker[3])), tag=tag))
            timer.add_send_time()
    for iw, walker in enumerate(data):
        if walker[1] == 0:
            timer.start_time()
            tag = walker[3] * walkers.nwalkers + comm.rank
            timer.add_non_communication()
            timer.start_time()
            comm.Recv(walkers.walker_buffer, source=int(round(walker[3])), tag=tag)
            timer.add_recv_time()
            timer.start_time()
            set_buffer(walkers, iw, walkers.walker_buffer)
            timer.add_non_communication()
    timer.start_time()
    for r in reqs:
        r.wait()
    timer.add_communication()


def stochastic_reconfiguration(walkers, comm, timer=PopControllerTimer()):
    # gather all walker information on the root
    timer.start_time()
    nwalkers = walkers.nwalkers
    local_buffer = xp.array([get_buffer(walkers, i) for i in range(nwalkers)])
    walker_len = local_buffer[0].shape[0]
    global_buffer = None
    if comm.rank == 0:
        global_buffer = numpy.zeros((comm.size, nwalkers, walker_len), dtype=numpy.complex128)
    timer.add_non_communication()

    timer.start_time()
    comm.Gather(local_buffer, global_buffer, root=0)
    timer.add_communication()

    # perform sr on the root
    new_global_buffer = None
    timer.start_time()
    if comm.rank == 0:
        new_global_buffer = numpy.zeros((comm.size, nwalkers, walker_len), dtype=numpy.complex128)
        cumulative_weights = numpy.cumsum(abs(global_buffer[:, :, 0]))
        total_weight = cumulative_weights[-1]
        new_average_weight = total_weight / nwalkers / comm.size
        zeta = numpy.random.rand()
        for i in range(comm.size * nwalkers):
            z = (i + zeta) / nwalkers / comm.size
            new_i = numpy.searchsorted(cumulative_weights, z * total_weight)
            new_global_buffer[i // nwalkers, i % nwalkers] = global_buffer[
                new_i // nwalkers, new_i % nwalkers
            ]
            new_global_buffer[i // nwalkers, i % nwalkers, 0] = new_average_weight

    timer.add_non_communication()

    # distribute information of newly selected walkers
    timer.start_time()
    comm.Scatter(new_global_buffer, local_buffer, root=0)
    timer.add_communication()

    # set walkers using distributed information
    timer.start_time()
    for i in range(nwalkers):
        set_buffer(walkers, i, local_buffer[i])
    timer.add_non_communication()
