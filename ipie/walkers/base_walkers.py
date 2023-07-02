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
import time
from abc import ABCMeta, abstractmethod

import h5py
import numpy

from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import to_host
from ipie.utils.io import format_fixed_width_floats


class WalkerAccumulator:
    """Small class to handle passing around walker state."""

    def __init__(self, names, nsteps):
        self.names = names
        self.size = len(names)
        self.buffer = numpy.zeros((self.size,), dtype=numpy.complex128)
        self._data_index = {k: i for i, k in enumerate(self.names)}
        self.nsteps_per_block = nsteps
        self._eshift = 0.0

    def update(self, walkers):
        self.buffer += numpy.array(
            [
                to_host(xp.sum(walkers.weight)),
                to_host(xp.sum(walkers.unscaled_weight)),
                to_host(xp.sum(walkers.weight * walkers.hybrid_energy)),
            ]
        )

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
        nume = self.get_index("HybridEnergy")
        deno = self.get_index("Weight")
        vals[nume] = vals[nume] / vals[deno]
        vals[deno] = vals[deno] / factor
        ix = self.get_index("WeightFactor")
        vals[ix] = vals[ix] / factor

    def to_text(self, vals):
        return format_fixed_width_floats(vals.real)


class BaseWalkers(metaclass=ABCMeta):
    """Container for groups of walkers which make up a wavefunction.

    Parameters
    ----------
    system : object
        System object.
    nwalkers : int
        Number of walkers to initialise.
    """

    def __init__(
        self,
        nwalkers,
        verbose=False,
    ):
        self.nwalkers = nwalkers

        if verbose:
            print("# Setting up BaseWalkers.")
            print(f"# nwalkers = {self.nwalkers}")

        self.weight = numpy.array(
            [1.0 for iw in range(self.nwalkers)]  # TODO: allow for arbitrary initial weights
        )
        self.unscaled_weight = self.weight.copy()
        self.phase = numpy.array([1.0 + 0.0j for iw in range(self.nwalkers)])

        self.ovlp = numpy.array([1.0 for iw in range(self.nwalkers)])
        self.sgn_ovlp = numpy.array([1.0 for iw in range(self.nwalkers)])
        self.log_ovlp = numpy.array([0.0 for iw in range(self.nwalkers)])

        # in case we use local energy approximation to the propagation
        self.eloc = numpy.array([0.0 for iw in range(self.nwalkers)])

        self.hybrid_energy = numpy.array([0.0 for iw in range(self.nwalkers)])
        self.detR = [1.0 for iw in range(self.nwalkers)]
        self.detR_shift = numpy.array([0.0 for iw in range(self.nwalkers)])
        self.log_detR = [0.0 for iw in range(self.nwalkers)]
        self.log_shift = numpy.array([0.0 for iw in range(self.nwalkers)])
        self.log_detR_shift = [0.0 for iw in range(self.nwalkers)]

        self.buff_names = [
            "weight",
            "unscaled_weight",
            "phase",
            "hybrid_energy",
            "ovlp",
            "sgn_ovlp",
            "log_ovlp",
        ]
        self.buff_size = None
        self.walker_buffer = None
        self.write_file = None
        self.read_file = None

        if verbose:
            print("# Finish setting up walkers.handler.Walkers.")

    def set_buff_size_single_walker(self):
        names = []
        size = 0
        for k, v in self.__dict__.items():
            if not (k in self.buff_names):
                continue
            if isinstance(v, (xp.ndarray, numpy.ndarray)):
                names.append(k)
                size += v.size
            elif isinstance(v, (int, float, complex)):
                names.append(k)
                size += 1
            elif isinstance(v, list):
                names.append(k)
                for l in v:
                    if isinstance(l, (xp.ndarray)):
                        size += l.size
                    elif isinstance(l, (int, float, complex)):
                        size += 1
        return size

    def orthogonalise(self, free_projection=False):
        """Orthogonalise all walkers.

        Parameters
        ----------
        free_projection : bool
            True if doing free projection.
        """
        detR = self.reortho()
        if free_projection:
            (magn, dtheta) = cmath.polar(self.detR)
            self.weight *= magn
            self.phase *= cmath.exp(1j * dtheta)
        return detR

    def get_write_buffer(self):
        buff = numpy.concatenate(
            [
                [self.weight],
                [self.phase],
                [self.ovlp],
                self.phi.ravel(),
            ]
        )
        return buff

    def set_walkers_from_buffer(self, buff):
        self.weight = buff[0 : self.nwalkers]
        self.phase = buff[self.nwalkers : self.nwalkers * 2]
        self.ovlp = buff[self.nwalkers * 2 : self.nwalkers * 3]
        self.phi = buff[self.nwalkers * 3 :].reshape(self.phi.shape)

    def write_walkers_batch(self, comm):
        start = time.time()
        assert self.write_file is not None
        raise NotImplementedError("This is not tested. Please implement.")
        with h5py.File(self.write_file, "r+", driver="mpio", comm=comm) as fh5:
            # for (i,w) in enumerate(self.walkers):
            # ix = i + self.nwalkers*comm.rank
            fh5["walker_%d" % comm.rank][:] = self.get_write_buffer()
        if comm.rank == 0:
            print(" # Writing walkers to file.")
            print(f" # Time to write restart: {time.time() - start:13.8e} s")

    def read_walkers_batch(self, comm):
        assert self.write_file is not None
        with h5py.File(self.read_file, "r") as fh5:
            try:
                self.set_walkers_from_buffer(fh5["walker_%d" % comm.rank][:])
            except KeyError:
                print(f" # Could not read walker data from: {self.read_file}")

    @abstractmethod
    def reortho(self):
        pass

    @abstractmethod
    def reortho_batched(self):  # gpu version
        pass
