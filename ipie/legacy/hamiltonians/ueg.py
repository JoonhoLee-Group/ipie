import math
import sys
import time

import numpy
import scipy.linalg
import scipy.sparse

import ipie.utils
from ipie.utils.io import write_qmcpack_sparse


class UEG(object):
    """UEG system class (integrals read from fcidump)
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

    def __init__(self, system, options, verbose=False):
        if verbose:
            print("# Parsing input options.")
        self.name = "UEG"
        self.mu = options.get("mu", None)
        if self.mu is not None:
            print("# mu: {:6.4e}".format(self.mu))
        self._alt_convention = options.get("alt_convention", False)
        self.sparse = True

        # core energy
        self.ecore = 0.5 * system.ne * self.madelung(system)
        self.diagH1 = True

        skip_cholesky = options.get("skip_cholesky", False)

        self.kfac = system.kfac
        self.vol = system.vol
        # Single particle eigenvalues and corresponding kvectors
        (self.sp_eigv, self.basis, self.nmax) = self.sp_energies(
            system.ktwist, system.kfac, system.ecut
        )

        self.shifted_nmax = 2 * self.nmax
        self.imax_sq = numpy.dot(self.basis[-1], self.basis[-1])
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
        (eigs, qvecs, self.qnmax) = self.sp_energies(
            system.ktwist, system.kfac, 4 * system.ecut
        )
        # Omit Q = 0 term.
        self.qvecs = numpy.copy(qvecs[1:])
        self.vqvec = numpy.array([self.vq(system.kfac * q) for q in self.qvecs])
        # Number of momentum transfer vectors / auxiliary fields.
        # Can reduce by symmetry but be stupid for the moment.
        self.nchol = len(self.qvecs)
        self.nfields = 2 * len(self.qvecs)
        if verbose:
            print("# Number of plane waves: {:d}".format(self.nbasis))
            print("# Number of Cholesky vectors: {:d}.".format(self.nchol))
        # For consistency with frozen core molecular code.
        self.orbs = None
        self.frozen_core = False
        T = numpy.diag(self.sp_eigv)
        self.H1 = numpy.array([T, T])  # Making alpha and beta

        if skip_cholesky == False:
            h1e_mod = self.mod_one_body(T, system)
            self.h1e_mod = numpy.array([h1e_mod, h1e_mod])
        self.orbs = None
        self._opt = True

        nlimit = system.nup

        if system.thermal:
            nlimit = self.nbasis

        self.ikpq_i = []
        self.ikpq_kpq = []
        for (iq, q) in enumerate(self.qvecs):
            idxkpq_list_i = []
            idxkpq_list_kpq = []
            for i, k in enumerate(self.basis[0:nlimit]):
                kpq = k + q
                idxkpq = self.lookup_basis(kpq)
                if idxkpq is not None:
                    idxkpq_list_i += [i]
                    idxkpq_list_kpq += [idxkpq]
            self.ikpq_i += [idxkpq_list_i]
            self.ikpq_kpq += [idxkpq_list_kpq]

        self.ipmq_i = []
        self.ipmq_pmq = []
        for (iq, q) in enumerate(self.qvecs):
            idxpmq_list_i = []
            idxpmq_list_pmq = []
            for i, p in enumerate(self.basis[0:nlimit]):
                pmq = p - q
                idxpmq = self.lookup_basis(pmq)
                if idxpmq is not None:
                    idxpmq_list_i += [i]
                    idxpmq_list_pmq += [idxpmq]
            self.ipmq_i += [idxpmq_list_i]
            self.ipmq_pmq += [idxpmq_list_pmq]

        for (iq, q) in enumerate(self.qvecs):
            self.ikpq_i[iq] = numpy.array(self.ikpq_i[iq], dtype=numpy.int64)
            self.ikpq_kpq[iq] = numpy.array(self.ikpq_kpq[iq], dtype=numpy.int64)
            self.ipmq_i[iq] = numpy.array(self.ipmq_i[iq], dtype=numpy.int64)
            self.ipmq_pmq[iq] = numpy.array(self.ipmq_pmq[iq], dtype=numpy.int64)

        if skip_cholesky == False:
            if verbose:
                print("# Constructing two-body potentials incore.")
            (self.chol_vecs, self.iA, self.iB) = self.two_body_potentials_incore()
            write_ints = options.get("write_integrals", None)
            if write_ints is not None:
                self.write_integrals(system)
            if verbose:
                print(
                    "# Approximate memory required for "
                    "two-body potentials: {:13.8e} GB.".format(
                        (3 * self.iA.nnz * 16 / (1024**3))
                    )
                )
                print("# Finished constructing two-body potentials.")
                print("# Finished setting up UEG system object.")

    def sp_energies(self, ks, kfac, ecut):
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
        nmax = int(math.ceil(numpy.sqrt((2 * ecut))))

        spval = []
        vec = []
        kval = []

        for ni in range(-nmax, nmax + 1):
            for nj in range(-nmax, nmax + 1):
                for nk in range(-nmax, nmax + 1):
                    spe = 0.5 * (ni**2 + nj**2 + nk**2)
                    if spe <= ecut:
                        kijk = [ni, nj, nk]
                        kval.append(kijk)
                        # Reintroduce 2 \pi / L factor.
                        ek = 0.5 * numpy.dot(
                            numpy.array(kijk) + ks, numpy.array(kijk) + ks
                        )
                        spval.append(kfac**2 * ek)

        # Sort the arrays in terms of increasing energy.
        spval = numpy.array(spval)
        ix = numpy.argsort(spval, kind="mergesort")
        spval = spval[ix]
        kval = numpy.array(kval)[ix]

        return (spval, kval, nmax)

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

    def madelung(self, system):
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
        return c1 * c2 / (system.ne ** (1.0 / 3.0) * system.rs)

    def vq(self, q):
        """The typical 3D Coulomb kernel
        Parameters
        ----------
        q : float
            a plane-wave vector
        Returns
        -------
        v_M: float
            3D Coulomb kernel (in Hartrees)
        """
        return 4 * math.pi / numpy.dot(q, q)

    def mod_one_body(self, T, system):
        """Add a diagonal term of two-body Hamiltonian to the one-body term
        Parameters
        ----------
        T : float
            one-body Hamiltonian (i.e. kinetic energy)
        Returns
        -------
        h1e_mod: float
            modified one-body Hamiltonian
        """
        h1e_mod = numpy.copy(T)

        fac = 1.0 / (2.0 * system.vol)
        for (i, ki) in enumerate(self.basis):
            for (j, kj) in enumerate(self.basis):
                if i != j:
                    q = self.kfac * (ki - kj)
                    h1e_mod[i, i] = h1e_mod[i, i] - fac * self.vq(q)
        return h1e_mod

    def density_operator(self, iq):
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
        nnz = self.rho_ikpq_kpq[iq].shape[0]  # Number of non-zeros
        ones = numpy.ones((nnz), dtype=numpy.complex128)
        rho_q = scipy.sparse.csc_matrix(
            (ones, (self.rho_ikpq_kpq[iq], self.rho_ikpq_i[iq])),
            shape=(self.nbasis, self.nbasis),
            dtype=numpy.complex128,
        )
        return rho_q

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

    def write_integrals(self, system, filename="hamil.h5"):
        write_qmcpack_sparse(
            self.H1[0],
            2 * self.chol_vecs.toarray(),
            system.nelec,
            self.nbasis,
            enuc=0.0,
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
            return 1.0 / self.vol * self.vq(self.kfac * q1)
        else:
            return 0.0

    def compute_real_transformation(self):
        U22 = numpy.zeros((2, 2), dtype=numpy.complex128)
        U22[0, 0] = 1.0 / numpy.sqrt(2.0)
        U22[0, 1] = 1.0 / numpy.sqrt(2.0)
        U22[1, 0] = -1.0j / numpy.sqrt(2.0)
        U22[1, 1] = 1.0j / numpy.sqrt(2.0)

        U = numpy.zeros((self.nbasis, self.nbasis), dtype=numpy.complex128)

        for i, b in enumerate(self.basis):
            if numpy.sum(b * b) == 0:
                U[i, i] = 1.0
            else:
                mb = -b
                diff = numpy.einsum("ij->i", (self.basis - mb) ** 2)
                idx = numpy.argwhere(diff == 0)
                assert idx.ravel().shape[0] == 1
                if i < idx:
                    idx = idx.ravel()[0]
                    U[i, i] = U22[0, 0]
                    U[i, idx] = U22[0, 1]
                    U[idx, i] = U22[1, 0]
                    U[idx, idx] = U22[1, 1]
                else:
                    continue

        U = U.T.copy()
        return U

    def eri_4(self):
        eri_chol = 4 * self.chol_vecs.dot(self.chol_vecs.T)
        eri_chol = (
            eri_chol.toarray()
            .reshape((self.nbasis, self.nbasis, self.nbasis, self.nbasis))
            .real
        )
        eri_chol = eri_chol.transpose(0, 1, 3, 2)
        return eri_chol

    # Compute 8-fold symmetric integrals. Useful for running standard quantum chemistry methods
    def eri_8(self):

        eri = self.eri_4()
        U = self.compute_real_transformation()

        eri0 = numpy.einsum("mp,mnls->pnls", U.conj(), eri, optimize=True)
        eri1 = numpy.einsum("nq,pnls->pqls", U, eri0, optimize=True)
        eri2 = numpy.einsum("lr,pqls->pqrs", U.conj(), eri1, optimize=True)
        eri3 = numpy.einsum("st,pqrs->pqrt", U, eri2, optimize=True).real

        return eri3


if __name__ == "__main__":
    unit_test()
