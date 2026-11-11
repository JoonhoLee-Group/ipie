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
import plum
from numba import njit, prange

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import cast_to_device, qr, qr_mode, synchronize
from ipie.walkers.base_walkers import BaseWalkers
from ipie.walkers.uhf_walkers import UHFWalkers, invert_batched_overlap_cpu


@njit(parallel=True, cache=True)
def reortho_ghf_cpu(phi, log_det):
    """Walker-parallel QR: phi[w] <- Q diag(phase R_ii), log_det[w] = sum_i log|R_ii|.

    One walker per numba thread; the LAPACK calls inside must run single threaded.
    """
    for iw in prange(phi.shape[0]):
        q, r = numpy.linalg.qr(phi[iw])
        ld = 0.0
        for j in range(r.shape[0]):
            a = abs(r[j, j])
            s = r[j, j] / a if a > 0.0 else 0.0
            for i in range(q.shape[0]):
                phi[iw, i, j] = q[i, j] * s
            ld += numpy.log(a)
        log_det[iw] = ld


class GHFWalkers(BaseWalkers):
    """GHF style walker.

    Parameters
    ----------
    walkers : UHFWalkers
        UHFWalkers instance.
    initial_walker : :class:`numpy.ndarray`
        Initial GHF coefficient matrix.
    nup, ndown : int
        Number of spin up, down electrons.
    nbasis : int
        Number of basis functions.
    nwalkers : int
        Number of walkers.
    mpi_handler : MPIHandler
        MPIHandler instance.
    verbose : bool
        Verbosity.
    """

    @plum.dispatch
    def __init__(self, walkers: UHFWalkers, verbose: bool = False):
        self.nup = walkers.nup
        self.ndown = walkers.ndown
        self.nbasis = walkers.nbasis
        self.mpi_handler = walkers.mpi_handler
        super().__init__(walkers.nwalkers, verbose=verbose)

        self.phi = numpy.zeros(
            (self.nwalkers, self.nbasis * 2, self.nup + self.ndown), dtype=walkers.phia.dtype
        )

        # for iw in range(self.nwalkers):
        self.phi[:, : self.nbasis, : self.nup] = walkers.phia.copy()
        self.phi[:, self.nbasis :, self.nup :] = walkers.phib.copy()

        self.G = numpy.zeros(
            (self.nwalkers, 2 * self.nbasis, 2 * self.nbasis), dtype=walkers.Ga.dtype
        )
        self.G[:, : self.nbasis, : self.nbasis] = walkers.Ga.copy()
        self.G[:, self.nbasis :, self.nbasis :] = walkers.Gb.copy()
        self.Ghalf = None
        self.inv_ovlp = None

        self.rhf = None

        self.buff_names += ["phi"]
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    @property
    def phia(self):
        return self.phi[:, : self.nbasis, : self.nup]

    @phia.setter
    def phia(self, value):
        self.phi[:, : self.nbasis, : self.nup] = value

    @property
    def phib(self):
        return self.phi[:, self.nbasis :, self.nup :]

    @phib.setter
    def phib(self, value):
        self.phi[:, self.nbasis :, self.nup :] = value

    @plum.dispatch
    def __init__(
        self,
        initial_walker: numpy.ndarray,
        nup: int,
        ndown: int,
        nbasis: int,
        nwalkers: int,
        mpi_handler=None,
        verbose: bool = False,
    ):
        assert len(initial_walker.shape) == 2
        self.nup = nup
        self.ndown = ndown
        self.nbasis = nbasis
        self.mpi_handler = mpi_handler

        super().__init__(nwalkers, verbose=verbose)

        # should completely deprecate these
        self.field_configs = None

        self.phi = numpy.array(
            [initial_walker.copy() for iw in range(self.nwalkers)],
            dtype=numpy.complex128,
        )

        # will be built only on request
        self.G = numpy.zeros(
            shape=(self.nwalkers, 2 * self.nbasis, 2 * self.nbasis),
            dtype=numpy.complex128,
        )
        self.inv_ovlp = None

        self.rhf = None

        self.buff_names += ["phi"]
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    def build(self, trial):
        self.ovlp = trial.calc_greens_function(self)
        self.inverse_overlap(trial)
        if "inv_ovlp" not in self.buff_names:
            self.buff_names.append("inv_ovlp")
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = numpy.zeros(self.buff_size, dtype=numpy.complex128)

    def inverse_overlap(self, trial):
        ovlp = xp.matmul(trial.psi0.conj().T[None, :, :], self.phi)
        self.inv_ovlp = (
            xp.linalg.inv(ovlp)
            if config.get_option("use_gpu")
            else invert_batched_overlap_cpu(ovlp)
        )

    # This function casts relevant member variables into cupy arrays
    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose)

    def reortho(self):
        """reorthogonalise walkers."""
        if config.get_option("use_gpu"):
            return self.reortho_batched()
        # The phase of each R_ii goes into the orbitals so that det(R) stays real and positive;
        # det(R) = exp(sum_i log|R_ii| - detR_shift) avoids over/underflow.
        self.phi = numpy.ascontiguousarray(self.phi, dtype=numpy.complex128)
        log_det = numpy.zeros(self.nwalkers)
        reortho_ghf_cpu(self.phi, log_det)
        detR = numpy.exp(log_det - self.detR_shift)
        self.log_detR += numpy.log(detR)
        self.detR[:] = detR
        self.ovlp[:] = self.ovlp / detR
        return detR

    def reortho_batched(self):
        """reorthogonalise walkers."""
        assert config.get_option("use_gpu")
        self.phi, Rup = qr(self.phi, mode=qr_mode)
        Rup_diag = xp.einsum("wii->wi", Rup)
        # Move the phase of R_ii into the orbitals so that det(R) is real and positive and
        # ovlp / detR stays <psi_T|phi> (the Hubbard propagators carry the overlap forward).
        self.phi *= (Rup_diag / abs(Rup_diag))[:, None, :]
        log_det = xp.einsum("wi->w", xp.log(abs(Rup_diag)))
        self.detR = xp.exp(log_det - self.detR_shift)
        self.ovlp = self.ovlp / self.detR

        synchronize()

        return self.detR
