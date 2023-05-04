import numpy as np

from ipie.propagation.overlap import calc_overlap_multi_det
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.estimators.greens_function_multi_det import greens_function_multi_det
from ipie.trial_wavefunction.half_rotate import half_rotate_generic
from ipie.propagation.force_bias import construct_force_bias_batch_multi_det_trial


class NOCI(TrialWavefunctionBase):
    # Non-orthogonal MultiSlater
    def __init__(self, wavefunction, num_elec, num_basis, verbose=False):
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)
        assert len(wavefunction) == 2
        self.verbose = verbose
        self.num_elec = num_elec
        self.nbasis = num_basis
        self.nalpha, self.nbeta = self.num_elec
        slater_mats = wavefunction[1]
        self.psi = slater_mats
        self._num_dets = len(self.psi)
        self._max_num_dets = self.num_dets
        imag_norm = np.sum(self.psi.imag.ravel() * self.psi.imag.ravel())
        if imag_norm <= 1e-8:
            self.psi = np.array(self.psi.real, dtype=np.float64)
        self.coeffs = np.array(wavefunction[0], dtype=np.complex128)

        self.psia = self.psi[:, :, : self.nalpha]
        self.psib = self.psi[:, :, self.nalpha :]
        # self.G = self.compute_1rdm()

    def build(self) -> None:
        pass

    def calculate_energy(self, system, hamiltonian):
        pass
        # if self.verbose:
        # print("# Computing trial wavefunction energy.")
        # start = time.time()
        # if self.verbose:
        # print(
        # "# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
        # % (self.energy.real, self.e1b.real, self.e2b.real)
        # )
        # print("# Time to evaluate local energy: {} s".format(time.time() - start))

    def half_rotate(self, system, hamiltonian, comm=None):
        orbsa = self.psi[: self.num_dets, :, : self.nalpha]
        orbsb = self.psi[: self.num_dets, :, self.nalpha :]
        rot_1body, rot_chol = half_rotate_generic(
            self,
            system,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=self.num_dets,
            verbose=self.verbose,
        )
        self._rH1a = rot_1body[0]
        self._rH1b = rot_1body[1]
        self._rchola = rot_chol[0]
        self._rcholb = rot_chol[1]

    def calc_overlap(self, walkers) -> np.ndarray:
        return calc_overlap_multi_det(walkers, self)

    def calc_greens_function(self, walkers) -> np.ndarray:
        return greens_function_multi_det(walkers, self)
    
    def calc_force_bias(self, hamiltonian, walkers, mpi_handler=None) -> np.ndarray:
        return construct_force_bias_batch_multi_det_trial(hamiltonian, walkers, self)