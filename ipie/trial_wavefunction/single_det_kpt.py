import time
from typing import Optional, Tuple

import numpy
import plum

from ipie.config import CommType, config, MPI
from ipie.estimators.utils import gabk_spin, gabk_spin_nonuniform
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm, KptISDF
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.propagation.force_bias import (
    construct_force_bias_kpt_batch_single_det,
    construct_force_bias_kptsymm_batch_single_det,
    construct_force_bias_kptisdf_batch_single_det,
    construct_force_bias_kptsymm_batch_single_det_chunked,
)
from ipie.trial_wavefunction.half_rotate import (
    half_rotate_generic,
    half_rotate_chunked,
    half_rotate_isdf,
)
from ipie.propagation.overlap import calc_overlap_single_det_kpt
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.estimators.greens_function_kpt_single_det import (
    greens_function_kpt_single_det,
    greens_function_kpt_single_det_batch,
)
from ipie.utils.backend import arraylib as xp
from ipie.utils.mpi import MPIHandler
from typing import Union


# class for Single Determinant trial wavefunction for k-point calculations
class KptSingleDet(TrialWavefunctionBase):
    def __init__(
        self,
        wavefunction,
        nkpts,
        num_elec,
        num_basis,
        handler=MPIHandler(),
        noccas=None,
        noccbs=None,
        verbose=False,
    ):
        assert isinstance(wavefunction, numpy.ndarray)
        assert len(wavefunction.shape) == 3  # nkpts, nbasis, nocc
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        if verbose:
            print("# Parsing input options for trial_wavefunction.MultiSlater.")
        self.psi = wavefunction
        self.num_elec = num_elec
        self.nk = nkpts
        self._num_dets = 1
        self._max_num_dets = 1
        imag_norm = numpy.sum(self.psi.imag.ravel() * self.psi.imag.ravel())
        if imag_norm <= 1e-8:
            # print("# making trial wavefunction MO coefficient real")
            self.psi = numpy.array(self.psi.real, dtype=numpy.float64)

        self.psi0a = self.psi[:, :, : self.nalpha]
        self.psi0b = self.psi[:, :, self.nalpha :]
        if noccas is not None:
            self.noccas = noccas
            if noccbs is None:
                self.noccbs = noccas
                self.G, self.Ghalf = gabk_spin_nonuniform(
                    self.psi, self.psi, self.nalpha, self.nbeta, noccas, noccas
                )
            else:
                self.noccbs = noccbs
                self.G, self.Ghalf = gabk_spin_nonuniform(
                    self.psi, self.psi, self.nalpha, self.nbeta, noccas, noccbs
                )
        else:
            self.noccas = None
            self.noccbs = None

            self.G, self.Ghalf = gabk_spin(self.psi, self.psi, self.nalpha, self.nbeta)
        self.handler = handler

        self.psi0a = numpy.ascontiguousarray(self.psi0a)
        self.psi0b = numpy.ascontiguousarray(self.psi0b)

    def build(self) -> None:
        pass

    @property
    def num_dets(self) -> int:
        return 1

    @num_dets.setter
    def num_dets(self, ndets: int) -> None:
        raise RuntimeError("Cannot modify number of determinants in SingleDet trial.")

    def calculate_energy(self, system, hamiltonian) -> numpy.ndarray:
        pass

    @plum.dispatch
    def half_rotate(
        self: "KptSingleDet",
        hamiltonian: KptComplexChol,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nk, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nk, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_generic(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola = rot_chol[0][0]
        self._rcholb = rot_chol[1][0]
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "KptSingleDet",
        hamiltonian: KptComplexCholSymm,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nk, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nk, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_generic(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola = rot_chol[0][0]
        self._rcholb = rot_chol[1][0]
        self._rcholbara = rot_chol[2][0]
        self._rcholbarb = rot_chol[3][0]
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "KptSingleDet",
        hamiltonian: KptComplexCholChunked,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nk, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nk, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_chunked(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0] if self.nbeta > 0 else None

        self._rchola_chunk = rot_chol[0][0]
        self._rcholb_chunk = rot_chol[1][0] if self.nbeta > 0 else None
        self._rcholbara_chunk = rot_chol[2][0]
        self._rcholbarb_chunk = rot_chol[3][0] if self.nbeta > 0 else None
        self.half_rotated = True

    @plum.dispatch
    def half_rotate(
        self: "KptSingleDet",
        hamiltonian: KptISDF,
        comm: Optional[CommType] = MPI.COMM_WORLD,
    ):
        num_dets = 1
        orbsa = self.psi0a.reshape((num_dets, self.nk, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((num_dets, self.nk, self.nbasis, self.nbeta))
        rot_1body, rot_cgto = half_rotate_isdf(
            self,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=num_dets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rcgtoa = rot_cgto[0][0]
        self._rcgtob = rot_cgto[1][0]
        self.half_rotated = True

    def calc_overlap(self, walkers: "UHFWalkers") -> xp.ndarray:
        return calc_overlap_single_det_kpt(walkers, self)

    def calc_greens_function(self, walkers, build_full: bool = False) -> xp.ndarray:
        if config.get_option("use_gpu"):
            return greens_function_kpt_single_det_batch(walkers, self, build_full=build_full)
        else:
            return greens_function_kpt_single_det(walkers, self, build_full=build_full)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: KptComplexChol,
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> Tuple[xp.ndarray, xp.ndarray]:
        if hamiltonian.chunked:
            raise NotImplementedError
        else:
            return construct_force_bias_kpt_batch_single_det(hamiltonian, walkers, self)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: Union[KptComplexCholSymm, KptComplexCholChunked],
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> Tuple[xp.ndarray, xp.ndarray]:
        if hamiltonian.chunked:
            return construct_force_bias_kptsymm_batch_single_det_chunked(
                hamiltonian, walkers, self, mpi_handler
            )
        else:
            return construct_force_bias_kptsymm_batch_single_det(hamiltonian, walkers, self)

    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: KptISDF,
        walkers: UHFWalkers,
        mpi_handler: MPIHandler,
    ) -> Tuple[xp.ndarray, xp.ndarray]:
        if hamiltonian.chunked:
            raise NotImplementedError
        else:
            return construct_force_bias_kptisdf_batch_single_det(hamiltonian, walkers, self)
