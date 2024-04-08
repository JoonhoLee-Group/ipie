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
import scipy.linalg

from ipie.addons.thermal.estimators.particle_number import particle_number
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G
from ipie.addons.thermal.estimators.greens_function import greens_function_qr_strat
from ipie.addons.thermal.walkers.stack import PropagatorStack
from ipie.utils.misc import update_stack
from ipie.walkers.base_walkers import BaseWalkers
from ipie.addons.thermal.trial.one_body import OneBody

class UHFThermalWalkers(BaseWalkers):
    def __init__(
        self,
        trial: OneBody,
        nbasis: int,
        nwalkers: int,
        stack_size = None,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        mpi_handler = None,
        verbose: bool = False,
    ):
        """UHF style walker.
        """
        assert isinstance(trial, OneBody)
        super().__init__(nwalkers, verbose=verbose)

        self.nbasis = nbasis
        self.mpi_handler = mpi_handler
        self.nslice = trial.nslice
        self.stack_size = stack_size

        if self.stack_size == None:
            self.stack_size = trial.stack_size

        if (self.nslice // self.stack_size) * self.stack_size != self.nslice:
            if verbose:
                print("# Input stack size does not divide number of slices.")
            self.stack_size = update_stack(self.stack_size, self.nslice, verbose)

        if self.stack_size > trial.stack_size:
            if verbose:
                print("# Walker stack size differs from that estimated from trial density matrix.")
                print(f"# Be careful. cond(BT)**stack_size: {trial.cond ** self.stack_size:10.3e}.")

        self.stack_length = self.nslice // self.stack_size
        self.lowrank = lowrank
        self.lowrank_thresh = lowrank_thresh

        self.Ga = numpy.zeros(
                    shape=(self.nwalkers, self.nbasis, self.nbasis),
                    dtype=numpy.complex128)
        self.Gb = numpy.zeros(
                    shape=(self.nwalkers, self.nbasis, self.nbasis),
                    dtype=numpy.complex128)
        self.Ghalf = None

        max_diff_diag = numpy.linalg.norm(
                            (numpy.diag(
                                trial.dmat[0].diagonal()) - trial.dmat[0]))

        if max_diff_diag < 1e-10:
            self.diagonal_trial = True
            if verbose:
                print("# Trial density matrix is diagonal.")
        else:
            self.diagonal_trial = False
            if verbose:
                print("# Trial density matrix is not diagonal.")

        if verbose:
            print(f"# Walker stack size: {self.stack_size}")
            print(f"# Using low rank trick: {self.lowrank}")

        self.stack = [PropagatorStack(
            self.stack_size,
            self.nslice,
            self.nbasis,
            numpy.complex128,
            trial.dmat,
            trial.dmat_inv,
            diagonal=self.diagonal_trial,
            lowrank=self.lowrank,
            thresh=self.lowrank_thresh,
        ) for iw in range(self.nwalkers)]

        # Initialise all propagators to the trial density matrix.
        for iw in range(self.nwalkers):
            self.stack[iw].set_all(trial.dmat)
            greens_function_qr_strat(self, iw)
            self.stack[iw].G[0] = self.Ga[iw]
            self.stack[iw].G[1] = self.Gb[iw]
        
        # Shape (nwalkers,).
        self.M0a = numpy.array([
                    scipy.linalg.det(self.Ga[iw], check_finite=False) for iw in range(self.nwalkers)])
        self.M0b = numpy.array([
                    scipy.linalg.det(self.Gb[iw], check_finite=False) for iw in range(self.nwalkers)])

        for iw in range(self.nwalkers):
            self.stack[iw].ovlp = numpy.array([1.0 / self.M0a[iw], 1.0 / self.M0b[iw]])

        # # Temporary storage for stacks...
        # We should kill these here and store them in stack (10/02/2023)
        # I = numpy.identity(self.nbasis, dtype=numpy.complex128)
        # One = numpy.ones(self.nbasis, dtype=numpy.complex128)
        # self.Tl = numpy.array([I, I])
        # self.Ql = numpy.array([I, I])
        # self.Dl = numpy.array([One, One])
        # self.Tr = numpy.array([I, I])
        # self.Qr = numpy.array([I, I])
        # self.Dr = numpy.array([One, One])

        self.hybrid_energy = 0.0
        if verbose:
            for iw in range(self.nwalkers):
                G = numpy.array([self.Ga[iw], self.Gb[iw]])
                P = one_rdm_from_G(G)
                nav = particle_number(P)
                print(f"# Trial electron number for {iw}-th walker: {nav}")

        self.buff_names += ["Ga", "Gb"]
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    def calc_greens_function(self, iw, slice_ix=None, inplace=True):
        """Return the Green's function for walker `iw`.
        """
        if self.lowrank:
            return self.stack[iw].G # G[0] = Ga, G[1] = Gb

        else:
            return greens_function_qr_strat(self, iw, slice_ix=slice_ix, inplace=inplace)
    

    def reset(self, trial):
        self.weight = numpy.ones(self.nwalkers)
        self.phase = numpy.ones(self.nwalkers, dtype=numpy.complex128)

        for iw in range(self.nwalkers):
            self.stack[iw].reset()
            self.stack[iw].set_all(trial.dmat)
            self.calc_greens_function(iw)
    
    # For compatibiltiy with BaseWalkers class.
    def reortho(self):
        pass

    def reortho_batched(self):
        pass
