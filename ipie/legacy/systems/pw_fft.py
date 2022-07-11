import itertools
import math
import sys
import time

import numpy
import scipy.linalg
import scipy.sparse

import ipie.utils
# from ipie.utils.io import dump_qmcpack_cholesky
from ipie.legacy.trial_wavefunction.free_electron import FreeElectron

try:
    from ipie.legacy.estimators.ueg_kernels import mod_one_body, vq
except ImportError:
    print("ueg_kernels doesn't exist")
    pass


def fill_up_range(nmesh):
    a = numpy.zeros(nmesh)
    n = nmesh // 2
    a = numpy.linspace(-n, n, num=nmesh, dtype=numpy.int32)
    return a


class PW_FFT(object):
    """PW_FFT system class
    Parameters
    ----------
    nup : int
        Number of up electrons.
    ndown : int
        Number of down electrons.
    rs : float
        Density parameter.
    ecut : float
        Scaled cutoff energy.
    ktwist : :class:`numpy.ndarray`
        Twist vector.
    verbose : bool
        Print extra information.
    Attributes
    ----------
    T : :class:`numpy.ndarray`
        One-body part of the Hamiltonian. This is diagonal in plane wave basis.
    ecore : float
        Madelung contribution to the total energy.
    h1e_mod : :class:`numpy.ndarray`
        Modified one-body Hamiltonian.
    nfields : int
        Number of field configurations per walker for back propagation.
    basis : :class:`numpy.ndarray`
        Basis vectors within a cutoff.
    kfac : float
        Scale factor (2pi/L).
    """

    def __init__(self, inputs, verbose=False):
        if verbose:
            print("# Parsing input options.")
        self.name = "PW_FFT"
        print("# {}".format(self.name))
        self.nup = inputs.get("nup")
        self.ndown = inputs.get("ndown")
        self.nelec = (self.nup, self.ndown)
        self.rs = inputs.get("rs")
        self.ecut = inputs.get("ecut")
        self.ktwist = numpy.array(inputs.get("ktwist", [0, 0, 0])).reshape(3)
        self.mu = inputs.get("mu", None)
        if verbose:
            print("# Number of spin-up electrons: %i" % self.nup)
            print("# Number of spin-down electrons: %i" % self.ndown)
            print("# rs: %10.5f" % self.rs)

        self.thermal = inputs.get("thermal", False)
        self._alt_convention = inputs.get("alt_convention", False)

        self.diagH1 = inputs.get("diagonal_H1", True)

        # total # of electrons
        self.ne = self.nup + self.ndown
        # core energy
        self.ecore = 0.5 * self.ne * self.madelung()
        # spin polarisation
        self.zeta = (self.nup - self.ndown) / self.ne
        # Density.
        self.rho = ((4.0 * math.pi) / 3.0 * self.rs**3.0) ** (-1.0)
        # Box Length.
        self.L = self.rs * (4.0 * self.ne * math.pi / 3.0) ** (1 / 3.0)
        # Volume
        self.vol = self.L**3.0
        # k-space grid spacing.
        self.kfac = 2 * math.pi / self.L
        # Fermi Wavevector (infinite system).
        self.kf = (3 * (self.zeta + 1) * math.pi**2 * self.ne / self.L**3) ** (
            1 / 3.0
        )
        # Fermi energy (inifinite systems).
        self.ef = 0.5 * self.kf**2
        #

        skip_cholesky = inputs.get("skip_cholesky", False)

        if verbose:
            print("# Spin polarisation (zeta): %d" % self.zeta)
            print("# Electron density (rho): %13.8e" % self.rho)
            print("# Box Length (L): %13.8e" % self.L)
            print("# Volume: %13.8e" % self.vol)
            print("# k-space factor (2pi/L): %13.8e" % self.kfac)
            print("# Madelung Energy: %13.8e" % self.ecore)

        # Single particle eigenvalues and corresponding kvectors
        (self.sp_eigv, self.basis, self.nmax, self.gmap) = self.sp_energies(
            self.kfac, self.ecut
        )
        self.mesh = [self.nmax * 2 + 1] * 3

        self.shifted_nmax = 2 * self.nmax
        self.imax_sq = numpy.max(numpy.sum(self.basis * self.basis, axis=1))
        self.create_lookup_table()
        for (i, k) in enumerate(self.basis):
            assert i == self.lookup_basis(k)

        # Number of plane waves.
        self.nbasis = len(self.sp_eigv)
        self.nactive = self.nbasis
        self.ncore = 0
        self.nfv = 0
        self.mo_coeff = None

        # Allowed momentum transfers (4*ecut)
        (eigs, self.qvecs, self.qnmax, self.qmap) = self.sp_energies(
            self.kfac, 4 * self.ecut, nmax=self.nmax * 2
        )
        self.qmesh = [self.qnmax * 2 + 1] * 3
        self.vqvec = numpy.array([vq(self.kfac * q) for q in self.qvecs])
        self.sqrtvqvec = numpy.sqrt(self.vqvec)

        # Number of momentum transfer vectors / auxiliary fields.
        # Can reduce by symmetry but be stupid for the moment.
        self.nchol = len(self.qvecs)
        self.nfields = 2 * len(self.qvecs)
        if verbose:
            print("# Number of plane waves: %i" % self.nbasis)
            print("# Number of Cholesky vectors: %i" % self.nchol)

        # For consistency with frozen core molecular code.
        self.orbs = None
        self.frozen_core = False
        T = numpy.diag(self.sp_eigv)
        self.H1 = numpy.array([T, T])  # Making alpha and beta
        self.T = numpy.array([T, T])  # Making alpha and beta

        # if (skip_cholesky == False):
        h1e_mod = mod_one_body(T, numpy.asarray(self.basis), self.vol, self.kfac)
        self.h1e_mod = numpy.array([h1e_mod, h1e_mod])

        self.orbs = None
        self._opt = True

        sort_basis = numpy.argsort(numpy.diag(self.H1[0]), kind="mergesort")
        I = numpy.eye(self.nbasis)
        trial_a = I[:, sort_basis[: self.nup]].copy()
        trial_b = I[:, sort_basis[: self.ndown]].copy()

        # Hard coded to be RHF trial for now
        self.trial = numpy.zeros(
            (self.nbasis, self.nup + self.ndown), dtype=numpy.complex128
        )
        self.trial[:, : self.nup] = trial_a.copy()
        self.trial[:, self.nup :] = trial_b.copy()

        # if (skip_cholesky == False):
        #     if verbose:
        #         print("# Constructing two-body potentials incore.")
        #     (self.chol_vecs, self.iA, self.iB) = self.two_body_potentials_incore()
        #     write_ints = inputs.get('write_integrals', None)
        #     if write_ints is not None:
        #         self.write_integrals()
        #     if verbose:
        #         print("# Approximate memory required for "
        #               "two-body potentials: %f GB."%(3*self.iA.nnz*16/(1024**3)))
        #         print("# Constructing two_body_potentials_incore finished")
        #         print("# Finished setting up UEG system object.")

    def sp_energies(self, kfac, ecut, nmax=None):
        """Calculate the allowed kvectors and resulting single particle eigenvalues (basically kinetic energy)
        which can fit in the sphere in kspace determined by ecut.
        Parameters
        ----------
        kfac : float
            kspace grid spacing.
        ecut : float
            energy cutoff.
        Returns
        -------
        spval : :class:`numpy.ndarray`
            Array containing sorted single particle eigenvalues.
        kval : :class:`numpy.ndarray`
            Array containing basis vectors, sorted according to their
            corresponding single-particle energy.
        """

        # Scaled Units to match with HANDE.
        # So ecut is measured in units of 1/kfac^2.
        if nmax == None:
            nmax = int(math.ceil(numpy.sqrt((2 * ecut))))

        gx = fill_up_range(2 * nmax + 1)
        gy = fill_up_range(2 * nmax + 1)
        gz = fill_up_range(2 * nmax + 1)

        kall = numpy.array(list(itertools.product(*[gx, gy, gz])), dtype=numpy.int32)

        k2 = 0.5 * numpy.sum(kall * kall, axis=1)
        Gmap = numpy.argwhere(k2 <= ecut)
        Gmap = numpy.squeeze(Gmap)

        kval = kall[Gmap, :]

        kval_p_ktwist = kval + self.ktwist
        ek = 0.5 * numpy.sum(kval_p_ktwist * kval_p_ktwist, axis=1)
        spval = kfac**2 * ek

        return (spval, kval, nmax, Gmap)

    def madelung(self):
        """Use expression in Schoof et al. (PhysRevLett.115.130402) for the
        Madelung contribution to the total energy fitted to L.M. Fraser et al.
        Phys. Rev. B 53, 1814.
        Parameters
        ----------
        rs : float
            Wigner-Seitz radius.
        ne : int
            Number of electrons.
        Returns
        -------
        v_M: float
            Madelung potential (in Hartrees).
        """
        c1 = -2.837297
        c2 = (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)
        return c1 * c2 / (self.ne ** (1.0 / 3.0) * self.rs)

    def create_lookup_table(self):
        basis_ix = []
        for k in self.basis:
            basis_ix.append(self.map_basis_to_index(k))
        self.lookup = numpy.zeros(max(basis_ix) + 1, dtype=int)
        for i, b in enumerate(basis_ix):
            self.lookup[b] = i
        self.max_ix = max(basis_ix)

    def lookup_basis(self, vec):
        if numpy.dot(vec, vec) <= self.imax_sq:
            ix = self.map_basis_to_index(vec)
            if ix >= len(self.lookup):
                ib = None
            else:
                ib = self.lookup[ix]
            return ib
        else:
            ib = None

    def map_basis_to_index(self, k):
        return (
            (k[0] + self.nmax)
            + self.shifted_nmax * (k[1] + self.nmax)
            + self.shifted_nmax * self.shifted_nmax * (k[2] + self.nmax)
        )

    def scaled_density_operator_incore(self, transpose):
        """Density operator as defined in Eq.(6) of PRB(75)245123
        Parameters
        ----------
        q : float
            a plane-wave vector
        Returns
        -------
        rho_q: float
            density operator
        """
        rho_ikpq_i = []
        rho_ikpq_kpq = []
        for (iq, q) in enumerate(self.qvecs):
            idxkpq_list_i = []
            idxkpq_list_kpq = []
            for i, k in enumerate(self.basis):
                kpq = k + q
                idxkpq = self.lookup_basis(kpq)
                if idxkpq is not None:
                    idxkpq_list_i += [i]
                    idxkpq_list_kpq += [idxkpq]
            rho_ikpq_i += [idxkpq_list_i]
            rho_ikpq_kpq += [idxkpq_list_kpq]

        for (iq, q) in enumerate(self.qvecs):
            rho_ikpq_i[iq] = numpy.array(rho_ikpq_i[iq], dtype=numpy.int64)
            rho_ikpq_kpq[iq] = numpy.array(rho_ikpq_kpq[iq], dtype=numpy.int64)

        nq = len(self.qvecs)
        nnz = 0
        for iq in range(nq):
            nnz += rho_ikpq_kpq[iq].shape[0]

        col_index = []
        row_index = []

        values = []

        if transpose:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
                piovol = math.pi / (self.vol)
                if numpy.dot(qscaled, qscaled) < 1e-10:
                    factor = 0.0
                else:
                    factor = (piovol / numpy.dot(qscaled, qscaled)) ** 0.5

                for (innz, kpq) in enumerate(rho_ikpq_kpq[iq]):
                    row_index += [
                        rho_ikpq_kpq[iq][innz] + rho_ikpq_i[iq][innz] * self.nbasis
                    ]
                    col_index += [iq]
                    values += [factor]
        else:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
                piovol = math.pi / (self.vol)
                # factor = (piovol/numpy.dot(qscaled,qscaled))**0.5
                if numpy.dot(qscaled, qscaled) < 1e-10:
                    factor = 0.0
                else:
                    factor = (piovol / numpy.dot(qscaled, qscaled)) ** 0.5

                for (innz, kpq) in enumerate(rho_ikpq_kpq[iq]):
                    row_index += [
                        rho_ikpq_kpq[iq][innz] * self.nbasis + rho_ikpq_i[iq][innz]
                    ]
                    col_index += [iq]
                    values += [factor]

        rho_q = scipy.sparse.csc_matrix(
            (values, (row_index, col_index)),
            shape=(self.nbasis * self.nbasis, nq),
            dtype=numpy.complex128,
        )

        return rho_q

    def two_body_potentials_incore(self):
        """Calculatate A and B of Eq.(13) of PRB(75)245123 for a given plane-wave vector q
        Parameters
        ----------
        system :
            system class
        q : float
            a plane-wave vector
        Returns
        -------
        iA : numpy array
            Eq.(13a)
        iB : numpy array
            Eq.(13b)
        """
        # qscaled = self.kfac * self.qvecs

        # # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol

        rho_q = self.scaled_density_operator_incore(False)
        rho_qH = self.scaled_density_operator_incore(True)

        iA = 1j * (rho_q + rho_qH)
        iB = -(rho_q - rho_qH)

        return (rho_q, iA, iB)

    def write_integrals(self, filename="hamil.h5"):
        dump_qmcpack_cholesky(
            self.H1,
            2 * scipy.sparse.csr_matrix(self.chol_vecs),
            self.nelec,
            self.nbasis,
            e0=0.0,
            filename=filename,
        )

    def hijkl(self, i, j, k, l):
        """Compute <ij|kl> = (ik|jl) = 1/Omega * 4pi/(kk-ki)**2

        Checks for momentum conservation k_i + k_j = k_k + k_k, or
        k_k - k_i = k_j - k_l.

        Parameters
        ----------
        i, j, k, l : int
            Orbital indices for integral (ik|jl) = <ij|kl>.

        Returns
        -------
        integral : float
            (ik|jl)
        """
        q1 = self.basis[k] - self.basis[i]
        q2 = self.basis[j] - self.basis[l]
        if numpy.dot(q1, q1) > 1e-12 and numpy.dot(q1 - q2, q1 - q2) < 1e-12:
            return 1.0 / self.vol * vq(self.kfac * q1)
        else:
            return 0.0


# def unit_test():
#     from numpy import linalg as LA
#     from pyscf import gto, scf, ao2mo, mcscf, fci, ci, cc, tdscf, gw, hci
#     from ipie.legacy.systems.ueg import UEG
#     from ipie.legacy.trial_wavefunction.hartree_fock import HartreeFock
#     from ipie.legacy.trial_wavefunction.free_electron import FreeElectron

#     ecut = 1.0

#     inputs = {'nup':7,
#     'ndown':7,
#     'rs':1.0,
#     'thermal':False,
#     'ecut':ecut}

#     system = PW_FFT(inputs, True)
#     trial = FreeElectron(system, False, inputs, True)

#     from ipie.qmc.options import QMCOpts
#     from ipie.propagation.continuous import Continuous

#     system2 = UEG(inputs, True)

#     qmc = QMCOpts(inputs, system2, True)

#     trial = HartreeFock(system2, False, inputs, True)

# if __name__=="__main__":
#     unit_test()
