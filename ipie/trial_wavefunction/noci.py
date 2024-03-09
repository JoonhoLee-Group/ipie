import numpy as np
import scipy.linalg

from ipie.estimators.greens_function_multi_det import greens_function_noci
from ipie.estimators.local_energy import variational_energy_noci
from ipie.propagation.force_bias import construct_force_bias_batch_multi_det_trial
from ipie.propagation.overlap import calc_overlap_multi_det
from ipie.trial_wavefunction.half_rotate import half_rotate_generic
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase


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
        self.G = self.build_one_rdm()

    def build(self) -> None: ...

    def calculate_energy(self, system, hamiltonian):
        return variational_energy_noci(system, hamiltonian, self)

    def half_rotate(self, hamiltonian, comm=None):
        orbsa = self.psi[: self.num_dets, :, : self.nalpha]
        orbsb = self.psi[: self.num_dets, :, self.nalpha :]
        rot_1body, rot_chol = half_rotate_generic(
            self,
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

    def build_one_rdm(self):
        G = np.zeros((2, self.nbasis, self.nbasis), dtype=np.complex128)
        denom = 0.0j
        for ix, (detixa, detixb) in enumerate(zip(self.psia, self.psib)):
            c_ix = self.coeffs[ix]
            # <ix|
            for iy, (detiya, detiyb) in enumerate(zip(self.psia, self.psib)):
                # |iy>
                c_iy = self.coeffs[iy]
                # Matrix(<ix|iy>_a)
                omata = np.dot(detixa.T, detiya.conj())
                sign_a, logdet_a = np.linalg.slogdet(omata)
                # <ix|iy>_a
                deta = sign_a * np.exp(logdet_a)
                if deta < 1e-16:
                    continue
                # Matrix(<ix|iy>_b)
                omatb = np.dot(detixb.T, detiyb.conj())
                sign_b, logdet_b = np.linalg.slogdet(omatb)
                detb = sign_b * np.exp(logdet_b)
                if detb < 1e-16:
                    continue
                # <ix|iy>_a <ix|iy>_b
                ovlp = deta * detb
                # Matrix(<ix|iy>_a)^{-1}
                inv_ovlp = scipy.linalg.inv(omata)
                Ghalfa = np.dot(inv_ovlp, detixa.T)
                G[0] += c_ix.conj() * c_iy * ovlp * np.dot(detiya.conj(), Ghalfa)
                inv_ovlp = scipy.linalg.inv(omatb)
                Ghalfb = np.dot(inv_ovlp, detixb.T)
                G[1] += c_ix.conj() * c_iy * ovlp * np.dot(detiyb.conj(), Ghalfb)
                denom += c_ix.conj() * c_iy * ovlp
        return G / denom

    def calc_overlap(self, walkers) -> np.ndarray:
        return calc_overlap_multi_det(walkers, self)

    def calc_greens_function(self, walkers) -> np.ndarray:
        return greens_function_noci(walkers, self)

    def calc_force_bias(self, hamiltonian, walkers, mpi_handler=None) -> np.ndarray:
        return construct_force_bias_batch_multi_det_trial(hamiltonian, walkers, self)
