
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
#          Ankit Mahajan <ankitmahajan76@gmail.com>
#

import cmath
import copy
import math
import sys
import time

import h5py
import numpy
import scipy.linalg
from mpi4py import MPI

from ipie.utils.io import get_input_value, format_fixed_width_floats
from ipie.utils.misc import is_cupy, update_stack, to_numpy
from ipie.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from ipie.walkers.single_det_batch import SingleDetWalkerBatch

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host


class WalkerBatchHandler(object):
    """Container for groups of walkers which make up a wavefunction.

    Parameters
    ----------
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    nwalkers : int
        Number of walkers to initialise.
    nprop_tot : int
        Total number of propagators to store for back propagation + itcf.
    nbp : int
        Number of back propagation steps.
    """

    def __init__(
        self,
        system,
        hamiltonian,
        trial,
        qmc,
        walker_opts={},
        mpi_handler=None,
        nprop_tot=None,
        nbp=None,
        verbose=False,
    ):
        self.nwalkers = qmc.nwalkers
        self.ntot_walkers = qmc.ntot_walkers
        self.write_freq = walker_opts.get("write_freq", 0)
        self.write_file = walker_opts.get("write_file", "restart.h5")
        self.read_file = walker_opts.get("read_file", None)
        # weight, unscaled weight and hybrid energy accumulated across a block.
        # Mostly here for legacy purposes.
        self.accumulator_factors = WalkerAccumulator(
                ["Weight", "WeightFactor", "HybridEnergy"],
                qmc.nsteps
                )

        if mpi_handler is None:
            rank = 0
        else:
            rank = mpi_handler.comm.rank

        if verbose:
            print("# Setting up walkers.handler_batch.Walkers.")
            print("# qmc.nwalkers = {}".format(self.nwalkers))
            print("# qmc.ntot_walkers = {}".format(self.ntot_walkers))

        assert trial.name == "MultiSlater"

        if trial.ndets == 1:
            if verbose:
                print("# Using single det walker with a single det trial.")
            self.walker_type = "SD"
            if len(trial.psi.shape) == 3:
                trial.psi = trial.psi[0]
                trial.psia = trial.psia[0]
                trial.psib = trial.psib[0]
            self.walkers_batch = SingleDetWalkerBatch(
                system,
                hamiltonian,
                trial,
                nwalkers=self.nwalkers,
                walker_opts=walker_opts,
                index=0,
                nprop_tot=nprop_tot,
                nbp=nbp,
                mpi_handler=mpi_handler,
            )
        elif trial.ndets > 1:
            if verbose:
                print("# Using single det walker with a multi det trial.")
            self.walker_type = "SD"
            self.walkers_batch = MultiDetTrialWalkerBatch(
                system,
                hamiltonian,
                trial,
                nwalkers=self.nwalkers,
                walker_opts=walker_opts,
                index=0,
                nprop_tot=nprop_tot,
                nbp=nbp,
                mpi_handler=mpi_handler,
            )

        self.buff_size = self.walkers_batch.buff_size

        assert nbp == None

        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

        self.pcont_method = get_input_value(
            walker_opts,
            "population_control",
            default="pair_branch",
            alias=["pop_control"],
            verbose=verbose,
        )
        self.reconfiguration_counter = 0
        self.reconfiguration_freq = walker_opts.get("reconfiguration_freq", 50)
        self.min_weight = walker_opts.get("min_weight", 0.1)
        self.max_weight = walker_opts.get("max_weight", 4.0)

        if verbose:
            print(
                "# Using {} population control " "algorithm.".format(self.pcont_method)
            )
            mem = float(self.walker_buffer.nbytes) / (1024.0**3)
            print("# Buffer size for communication: {:13.8e} GB".format(mem))
            if mem > 2.0:
                # TODO: FDM FIX THIS
                print(
                    " # Warning: Walker buffer size > 2GB. May run into MPI" "issues."
                )

        if not self.walker_type == "thermal":
            walker_batch_size = 3 * self.nwalkers + self.walkers_batch.phia.size
            if not self.walkers_batch.rhf:
                walker_batch_size += self.walkers_batch.phib.size
        if self.write_freq > 0:
            self.write_restart = True
            self.dsets = []
            with h5py.File(
                self.write_file, "w", driver="mpio", comm=mpi_handler.comm
            ) as fh5:
                fh5.create_dataset(
                    "walker_batch_%d" % mpi.rank,
                    (walker_batch_size,),
                    dtype=numpy.complex128,
                )
        else:
            self.write_restart = False
        if self.read_file is not None:
            if verbose:
                print("# Reading walkers from %s file series." % self.read_file)
            self.read_walkers(mpi_handler.comm)

        self.target_weight = qmc.ntot_walkers
        # self.nw = qmc.nwalkers
        self.set_total_weight(qmc.ntot_walkers)
        self.start_time_const = 0.0
        self.communication_time = 0.0
        self.non_communication_time = 0.0
        self.recv_time = 0.0
        self.send_time = 0.0

        if verbose:
            print("# Finish setting up walkers.handler.Walkers.")

    def update_accumulators(self):
        self.accumulator_factors.update(self.walkers_batch)

    def zero_accumulators(self):
        self.accumulator_factors.zero()

    def orthogonalise(self, trial, free_projection):
        """Orthogonalise all walkers.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        free_projection : bool
            True if doing free projection.
        """
        detR = self.walkers_batch.reortho()
        if free_projection:
            (magn, dtheta) = cmath.polar(self.walkers_batch.detR)
            self.walkers_batch.weight *= magn
            self.walkers_batch.phase *= cmath.exp(1j * dtheta)


    def copy_historic_wfn(self):
        """Copy current wavefunction to psi_n for next back propagation step."""
        for (i, w) in enumerate(self.walkers):
            numpy.copyto(self.walkers[i].phi_old, self.walkers[i].phi)

    def copy_bp_wfn(self, phi_bp):
        """Copy back propagated wavefunction.

        Parameters
        ----------
        phi_bp : object
            list of walker objects containing back propagated walkers.
        """
        for (i, (w, wbp)) in enumerate(zip(self.walkers, phi_bp)):
            numpy.copyto(self.walkers[i].phi_bp, wbp.phi)

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

    def pop_control(self, comm):
        if is_cupy(self.walkers_batch.weight):
            import cupy

            array = cupy.asnumpy
        else:
            array = numpy.array

        self.start_time()
        if self.ntot_walkers == 1:
            return
        weights = numpy.abs(array(self.walkers_batch.weight))
        global_weights = numpy.empty(len(weights) * comm.size)
        self.add_non_communication()
        self.start_time()
        if self.pcont_method == "comb":
            comm.Allgather(weights, global_weights)
            total_weight = sum(global_weights)
        else:
            sum_weights = numpy.sum(weights)
            total_weight = numpy.empty(1, dtype=numpy.float64)
            comm.Reduce(sum_weights, total_weight, op=MPI.SUM, root=0)
            comm.Bcast(total_weight, root=0)
            total_weight = total_weight[0]

        self.add_communication()
        self.start_time()

        # Rescale weights to combat exponential decay/growth.
        scale = total_weight / self.target_weight
        if total_weight < 1e-8:
            if comm.rank == 0:
                print("# Warning: Total weight is {:13.8e}: ".format(total_weight))
                print("# Something is seriously wrong.")
            sys.exit()
        self.set_total_weight(total_weight)
        # Todo: Just standardise information we want to send between routines.
        self.walkers_batch.unscaled_weight = self.walkers_batch.weight
        self.walkers_batch.weight = self.walkers_batch.weight / scale
        if self.pcont_method == "comb":
            global_weights = global_weights / scale
            self.add_non_communication()
            self.comb(comm, global_weights)
        elif self.pcont_method == "pair_branch":
            # self.pair_branch(comm)
            self.pair_branch_fast(comm)
        elif self.pcont_method == "stochastic_reconfiguration":
            self.reconfiguration_counter += 1
            if self.reconfiguration_counter % self.reconfiguration_freq == 0:
                self.stochastic_reconfiguration(comm)
                self.reconfiguration_counter = 0
        else:
            if comm.rank == 0:
                print("Unknown population control method.")

    def comb(self, comm, weights):
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
        if is_cupy(self.walkers_batch.weight):
            import cupy

            array = cupy.asnumpy
        else:
            array = numpy.array

        self.start_time()
        if comm.rank == 0:
            parent_ix = numpy.zeros(len(weights), dtype="i")
        else:
            parent_ix = numpy.empty(len(weights), dtype="i")
        if comm.rank == 0:
            total_weight = sum(weights)
            cprobs = numpy.cumsum(weights)
            r = numpy.random.random()
            comb = [
                (i + r) * (total_weight / self.target_weight)
                for i in range(self.target_weight)
            ]
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

        self.add_non_communication()

        self.start_time()
        data = comm.bcast(data, root=0)
        self.add_communication()
        self.start_time()
        parent_ix = data["ix"]
        # Keep total weight saved for capping purposes.
        # where returns a tuple (array,), selecting first element.
        kill = numpy.where(parent_ix == 0)[0]
        clone = numpy.where(parent_ix > 1)[0]
        reqs = []
        walker_buffers = []
        # First initiate non-blocking sends of walkers.
        self.add_non_communication()
        self.start_time()
        comm.barrier()
        self.add_communication()
        for i, (c, k) in enumerate(zip(clone, kill)):
            # Sending from current processor?
            if c // self.nwalkers == comm.rank:
                self.start_time()
                # Location of walker to clone in local list.
                clone_pos = c % self.nwalkers
                # copying walker data to intermediate buffer to avoid issues
                # with accessing walker data during send. Might not be
                # necessary.
                dest_proc = k // self.nwalkers
                # with h5py.File('before_{}.h5'.format(comm.rank), 'a') as fh5:
                # fh5['walker_{}_{}_{}'.format(c,k,dest_proc)] = self.walkers[clone_pos].get_buffer()
                buff = self.walkers_batch.get_buffer(clone_pos)
                self.add_non_communication()
                self.start_time()
                reqs.append(comm.Isend(buff, dest=dest_proc, tag=i))
                self.add_send_time()
        # Now receive walkers on processors where walkers are to be killed.
        for i, (c, k) in enumerate(zip(clone, kill)):
            # Receiving to current processor?
            if k // self.nwalkers == comm.rank:
                self.start_time()
                # Processor we are receiving from.
                source_proc = c // self.nwalkers
                # Location of walker to kill in local list of walkers.
                kill_pos = k % self.nwalkers
                self.add_non_communication()
                self.start_time()
                comm.Recv(self.walker_buffer, source=source_proc, tag=i)
                # with h5py.File('walkers_recv.h5', 'w') as fh5:
                # fh5['walk_{}'.format(k)] = self.walker_buffer.copy()
                self.add_recv_time()
                self.start_time()
                self.walkers_batch.set_buffer(kill_pos, self.walker_buffer)
                self.add_non_communication()
                # with h5py.File('after_{}.h5'.format(comm.rank), 'a') as fh5:
                # fh5['walker_{}_{}_{}'.format(c,k,comm.rank)] = self.walkers[kill_pos].get_buffer()
        self.start_time()
        # Complete non-blocking send.
        for rs in reqs:
            rs.wait()
        # Necessary?
        # if len(kill) > 0 or len(clone) > 0:
        # sys.exit()
        comm.Barrier()
        self.add_communication()
        # Reset walker weight.
        # TODO: check this.
        # for w in self.walkers:
        # w.weight = 1.0
        self.start_time()
        self.walkers_batch.weight.fill(1.0)
        self.add_non_communication()

    def pair_branch_fast(self, comm):
        if is_cupy(self.walkers_batch.weight):
            import cupy

            abs = cupy.abs
            array = cupy.asnumpy
        else:
            abs = numpy.abs
            array = numpy.array

        self.start_time()
        walker_info_0 = array(abs(self.walkers_batch.weight))
        self.add_non_communication()

        self.start_time()
        glob_inf = None
        glob_inf_0 = None
        glob_inf_1 = None
        glob_inf_2 = None
        glob_inf_3 = None
        if comm.rank == 0:
            glob_inf_0 = numpy.empty(
                [comm.size, self.walkers_batch.nwalkers], dtype=numpy.float64
            )
            glob_inf_1 = numpy.empty(
                [comm.size, self.walkers_batch.nwalkers], dtype=numpy.int64
            )
            glob_inf_1.fill(1)
            glob_inf_2 = numpy.array(
                [
                    [r for i in range(self.walkers_batch.nwalkers)]
                    for r in range(comm.size)
                ],
                dtype=numpy.int64,
            )
            glob_inf_3 = numpy.array(
                [
                    [r for i in range(self.walkers_batch.nwalkers)]
                    for r in range(comm.size)
                ],
                dtype=numpy.int64,
            )

        self.add_non_communication()

        self.start_time()
        comm.Gather(walker_info_0, glob_inf_0, root=0)
        self.add_communication()

        # Want same random number seed used on all processors
        self.start_time()
        if comm.rank == 0:
            # Rescale weights.
            glob_inf = numpy.zeros(
                (self.walkers_batch.nwalkers * comm.size, 4), dtype=numpy.float64
            )
            glob_inf[:, 0] = glob_inf_0.ravel()
            glob_inf[:, 1] = glob_inf_1.ravel()
            glob_inf[:, 2] = glob_inf_2.ravel()
            glob_inf[:, 3] = glob_inf_3.ravel()
            total_weight = sum(w[0] for w in glob_inf)
            sort = numpy.argsort(glob_inf[:, 0], kind="mergesort")
            isort = numpy.argsort(sort, kind="mergesort")
            glob_inf = glob_inf[sort]
            s = 0
            e = len(glob_inf) - 1
            tags = []
            isend = 0
            while s < e:
                if glob_inf[s][0] < self.min_weight or glob_inf[e][0] > self.max_weight:
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
            nw = self.nwalkers
            glob_inf = glob_inf[isort].reshape((comm.size, nw, 4))
        else:
            data = None
            glob_inf = None
            total_weight = 0
        self.add_non_communication()
        self.start_time()

        data = numpy.empty([self.walkers_batch.nwalkers, 4], dtype=numpy.float64)
        comm.Scatter(glob_inf, data, root=0)

        self.add_communication()
        # Keep total weight saved for capping purposes.
        walker_buffers = []
        reqs = []
        for iw, walker in enumerate(data):
            if walker[1] > 1:
                self.start_time()
                tag = comm.rank * self.walkers_batch.nwalkers + walker[3]
                self.walkers_batch.weight[iw] = walker[0]
                buff = self.walkers_batch.get_buffer(iw)
                self.add_non_communication()
                self.start_time()
                reqs.append(comm.Isend(buff, dest=int(round(walker[3])), tag=tag))
                self.add_send_time()
        for iw, walker in enumerate(data):
            if walker[1] == 0:
                self.start_time()
                tag = walker[3] * self.walkers_batch.nwalkers + comm.rank
                self.add_non_communication()
                self.start_time()
                comm.Recv(self.walker_buffer, source=int(round(walker[3])), tag=tag)
                self.add_recv_time()
                self.start_time()
                self.walkers_batch.set_buffer(iw, self.walker_buffer)
                self.add_non_communication()
        self.start_time()
        for r in reqs:
            r.wait()
        self.add_communication()

    def pair_branch(self, comm):
        self.start_time()
        walker_info = [
            [abs(self.walkers_batch.weight[w]), 1, comm.rank, comm.rank]
            for w in range(self.walkers_batch.nwalkers)
        ]
        self.add_non_communication()
        self.start_time()
        glob_inf = comm.gather(walker_info, root=0)
        self.add_communication()

        # Want same random number seed used on all processors
        self.start_time()
        if comm.rank == 0:
            # Rescale weights.
            glob_inf = numpy.array([item for sub in glob_inf for item in sub])
            total_weight = sum(w[0] for w in glob_inf)
            sort = numpy.argsort(glob_inf[:, 0], kind="mergesort")
            isort = numpy.argsort(sort, kind="mergesort")
            glob_inf = glob_inf[sort]
            s = 0
            e = len(glob_inf) - 1
            tags = []
            isend = 0
            while s < e:
                if glob_inf[s][0] < self.min_weight or glob_inf[e][0] > self.max_weight:
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
            nw = self.nwalkers
            glob_inf = glob_inf[isort].reshape((comm.size, nw, 4))
        else:
            data = None
            total_weight = 0
        self.add_non_communication()
        self.start_time()
        data = comm.scatter(glob_inf, root=0)
        self.add_communication()
        # Keep total weight saved for capping purposes.
        walker_buffers = []
        reqs = []
        for iw, walker in enumerate(data):
            if walker[1] > 1:
                self.start_time()
                tag = comm.rank * len(walker_info) + walker[3]
                self.walkers_batch.weight[iw] = walker[0]
                buff = self.walkers_batch.get_buffer(iw)
                self.add_non_communication()
                self.start_time()
                reqs.append(comm.Isend(buff, dest=int(round(walker[3])), tag=tag))
                self.add_send_time()
        for iw, walker in enumerate(data):
            if walker[1] == 0:
                self.start_time()
                tag = walker[3] * len(walker_info) + comm.rank
                self.add_non_communication()
                self.start_time()
                comm.Recv(self.walker_buffer, source=int(round(walker[3])), tag=tag)
                self.add_recv_time()
                self.start_time()
                self.walkers_batch.set_buffer(iw, self.walker_buffer)
                self.add_non_communication()
        self.start_time()
        for r in reqs:
            r.wait()
        self.add_communication()

    def stochastic_reconfiguration(self, comm):
        if is_cupy(self.walkers_batch.weight):
            import cupy

            abs = cupy.abs
            array = cupy.asnumpy
        else:
            abs = numpy.abs
            array = numpy.array

        # gather all walker information on the root
        self.start_time()
        nwalkers = self.walkers_batch.nwalkers
        local_buffer = array(
            [self.walkers_batch.get_buffer(i) for i in range(nwalkers)]
        )
        walker_len = local_buffer[0].shape[0]
        global_buffer = None
        if comm.rank == 0:
            global_buffer = numpy.zeros(
                (comm.size, nwalkers, walker_len), dtype=numpy.complex128
            )
        self.add_non_communication()

        self.start_time()
        comm.Gather(local_buffer, global_buffer, root=0)
        self.add_communication()

        # perform sr on the root
        new_global_buffer = None
        self.start_time()
        if comm.rank == 0:
            new_global_buffer = numpy.zeros(
                (comm.size, nwalkers, walker_len), dtype=numpy.complex128
            )
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

        self.add_non_communication()

        # distribute information of newly selected walkers
        self.start_time()
        comm.Scatter(new_global_buffer, local_buffer, root=0)
        self.add_communication()

        # set walkers using distributed information
        self.start_time()
        for i in range(nwalkers):
            self.walkers_batch.set_buffer(i, local_buffer[i])
        self.add_non_communication()

    def set_total_weight(self, total_weight):
        self.walkers_batch.total_weight = total_weight

    def get_write_buffer(self):
        buff = numpy.concatenate(
            [
                [self.walkers_batch.weight],
                [self.walkers_batch.phase],
                [self.walkers_batch.ovlp],
                self.walkers_batch.phi.ravel(),
            ]
        )
        return buff

    def set_walkers_batch_from_buffer(self, buff):
        self.walkers_batch.weight = buff[0 : self.nwalkers]
        self.walkers_batch.phase = buff[self.nwalkers : self.nwalkers * 2]
        self.walkers_batch.ovlp = buff[self.nwalkers * 2 : self.nwalkers * 3]
        self.walkers_batch.phi = buff[self.nwalkers * 3 :].reshape(
            self.walkers_batch.phi.shape
        )

    def write_walkers_batch(self, comm):
        start = time.time()
        with h5py.File(self.write_file, "r+", driver="mpio", comm=comm) as fh5:
            # for (i,w) in enumerate(self.walkers):
            # ix = i + self.nwalkers*comm.rank
            buff = self.get_write_buffer()
            fh5["walker_%d" % comm.rank][:] = self.get_write_buffer()
        if comm.rank == 0:
            print(" # Writing walkers to file.")
            print(" # Time to write restart: {:13.8e} s".format(time.time() - start))

    def read_walkers_batch(self, comm):
        with h5py.File(self.read_file, "r") as fh5:
            try:
                self.set_walkers_batch_from_buffer(fh5["walker_%d" % comm.rank][:])
            except KeyError:
                print(" # Could not read walker data from:" " %s" % (self.read_file))

class WalkerAccumulator(object):
    """Small class to handle passing around walker state."""
    def __init__(self, names, nsteps):
        self.names = names
        self.size = len(names)
        self.buffer = numpy.zeros((self.size,), dtype=numpy.complex128)
        self._data_index = {k: i for i, k in enumerate(self.names)}
        self.nsteps_per_block = nsteps
        self._eshift = 0.0

    def update(self, walker_batch):
        self.buffer += numpy.array([
                    to_host(xp.sum(walker_batch.weight)),
                    to_host(xp.sum(walker_batch.unscaled_weight)),
                    to_host(xp.sum(walker_batch.weight*walker_batch.hybrid_energy))
                    ])

    def zero(self):
        self.buffer.fill(0.0j)

    def get_index(self, name):
        index = self._data_index.get(name, None)
        if index is None:
            raise RuntimeError(f"Unknown walker property {name}")
        return index

    @property
    def eshift(self):
        return self._eshift.real

    @eshift.setter
    def eshift(self, value):
        self._eshift = value

    def post_reduce_hook(self, vals, block):
        assert len(vals) == len(self.names)
        if block == 0:
            factor = 1
        else:
            factor = self.nsteps_per_block
        nume = self.get_index('HybridEnergy')
        deno = self.get_index('Weight')
        vals[nume] = vals[nume] / vals[deno]
        vals[deno] = vals[deno] / factor
        ix = self.get_index('WeightFactor')
        vals[ix] = vals[ix] / factor

    def to_text(self, vals):
        return format_fixed_width_floats(vals.real)
