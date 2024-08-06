import time
from typing import Tuple, Union

import numpy
import plum

from ipie.systems.generic import Generic
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.estimators.generic import cholesky_jk_ghf
from ipie.estimators.greens_function_single_det import greens_function_single_det_ghf
from ipie.estimators.utils import gab_mod
from ipie.propagation.overlap import calc_overlap_single_det_ghf
from ipie.propagation.force_bias import construct_force_bias_batch_single_det


class SingleDetGHF(TrialWavefunctionBase):
    # `num_basis`is # of spin-less AOs.
    @plum.dispatch
    def __init__(self, uhf_trial: Union[SingleDet, ParticleHole], verbose: bool = False):
        psi = numpy.hstack([uhf_trial.psi0a, uhf_trial.psi0b])
        nalpha = uhf_trial.psi0a.shape[-1]
        nbeta = uhf_trial.psi0b.shape[-1]
        nbasis = uhf_trial.psi0b.shape[0]
        super().__init__(psi, (nalpha, nbeta), nbasis, verbose=verbose)
        self.num_elec = (nalpha, nbeta)
        self._num_dets = 1
        self._max_num_dets = 1

        self.nocc = nalpha + nbeta
        self.psi0 = numpy.zeros((self.nbasis * 2, self.nocc), dtype=uhf_trial.psi0a.dtype)

        self.psi0[: self.nbasis, : self.nalpha] = uhf_trial.psi0a.copy()
        self.psi0[self.nbasis :, self.nalpha :] = uhf_trial.psi0b.copy()

        self.psi0a = self.psi0[: self.nbasis, : self.nocc]
        self.psi0b = self.psi0[self.nbasis :, : self.nocc]

        self.G = numpy.zeros((2*self.nbasis, 2*self.nbasis), dtype=uhf_trial.G[0].dtype)
        self.G[:self.nbasis, :self.nbasis] = uhf_trial.G[0].copy()
        self.G[self.nbasis:, self.nbasis:] = uhf_trial.G[1].copy()

    @plum.dispatch
    def __init__(
        self,
        wavefunction: numpy.ndarray,
        num_elec: Tuple[int, int],
        num_basis: int,
        verbose: bool = False,
    ):
        assert len(wavefunction.shape) == 2
        assert wavefunction.shape[0] // 2 == num_basis
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        if verbose:
            print("# Parsing input options for trial_wavefunction.SingleDetGHF.")
        self.psi = wavefunction
        self.num_elec = num_elec
        self._num_dets = 1
        self._max_num_dets = 1
        imag_norm = numpy.sum(self.psi.imag.ravel() * self.psi.imag.ravel())
        if imag_norm <= 1e-8:
            # print("# making trial wavefunction MO coefficient real")
            self.psi = numpy.array(self.psi.real, dtype=numpy.float64)

        self.nocc = self.nalpha + self.nbeta
        self.psi0 = self.psi[:, : self.nocc]

        # Can split alpha/beta part of the GHF wfn.
        self.psi0a = self.psi[:self.nbasis, :self.nocc]
        self.psi0b = self.psi[self.nbasis:, :self.nocc]

        self.G, self.Ghalf = gab_mod(self.psi, self.psi)

    def build(self) -> None:
        pass

    @property
    def num_dets(self) -> int:
        return 1

    @num_dets.setter
    def num_dets(self, ndets: int) -> None:
        raise RuntimeError("Cannot modify number of determinants in SingleDet trial.")

    @plum.dispatch
    def calculate_energy(
            self, system: Generic, hamiltonian: Union[GenericRealChol, GenericComplexChol]) -> None:
        if self.verbose:
            print("# Computing trial wavefunction energy.")
        start = time.time()
        nbasis = self.nbasis
        Gaa = self.G[:nbasis, :nbasis].copy()
        Gbb = self.G[nbasis:, nbasis:].copy()

        self.e1b = (
            numpy.sum(Gaa * hamiltonian.H1[0])
            + numpy.sum(Gbb * hamiltonian.H1[1])
            + hamiltonian.ecore
        )
        self.ej, self.ek = cholesky_jk_ghf(hamiltonian, self.G)
        self.e2b = self.ej - self.ek
        self.energy = self.e1b + self.e2b

        if self.verbose:
            print(
                f"# (E, E1B, E2B): ({self.energy.real:13.8e}, {self.e1b.real:13.8e},"
                f"{self.e2b.real:13.8e})"
            )
            print(f"# Time to evaluate trial energy: {time.time() - start}")

    @plum.dispatch
    def half_rotate(self, hamiltonian: GenericRealChol, comm):
        print("# Half rotation not implemented for GHF with GenericRealChol")
        print("# Will use full greens function")
        # Half rotation does not quite make sense for spin-split integrals.
        # one can do completely spin-orbital based implementation in the future.

    def calc_overlap(self, walkers) -> numpy.ndarray:
        return calc_overlap_single_det_ghf(walkers, self)

    def calc_greens_function(self, walkers, build_full: bool = False) -> numpy.ndarray:
        return greens_function_single_det_ghf(walkers, self)

    @plum.dispatch
    def calc_force_bias(
        self, 
        hamiltonian: GenericRealChol, 
        walkers: GHFWalkers, 
        mpi_handler = None,
    ) -> numpy.ndarray:
        return construct_force_bias_batch_single_det(hamiltonian, walkers)
    
    # TODO: check.
    @plum.dispatch
    def calc_force_bias(
        self,
        hamiltonian: GenericComplexChol,
        walkers: GHFWalkers,
        mpi_handler = None,
    ) -> numpy.ndarray:
        return construct_force_bias_batch_single_det(hamiltonian, walkers)
