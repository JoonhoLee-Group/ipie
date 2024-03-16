import numpy
import scipy.sparse
from ipie.utils.io import write_qmcpack_sparse

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
from ipie.addons.thermal.utils.ueg_kernels import (
    build_J_opt,
    build_K_opt,
    coulomb_greens_function_per_qvec,
    exchange_greens_function_per_qvec,
)


def exchange_greens_function(nq, kpq_i, kpq, pmq_i, pmq, Gprod, G):
    for iq in range(nq):
        for idxkpq, i in zip(kpq[iq], kpq_i[iq]):
            for idxpmq, j in zip(pmq[iq], pmq_i[iq]):
                Gprod[iq] += G[j, idxkpq] * G[i, idxpmq]


def coulomb_greens_function(nq, kpq_i, kpq, pmq_i, pmq, Gkpq, Gpmq, G):
    for iq in range(nq):
        for idxkpq, i in zip(kpq[iq], kpq_i[iq]):
            Gkpq[iq] += G[i, idxkpq]
        for idxpmq, i in zip(pmq[iq], pmq_i[iq]):
            Gpmq[iq] += G[i, idxpmq]


def local_energy_ueg(system, ham, G, Ghalf=None, two_rdm=None, debug=False):
    """Local energy computation for uniform electron gas
    Parameters
    ----------
    system :
        system class
    ham :
        hamiltonian class
    G :
        Green's function
    Returns
    -------
    etot : float
        total energy
    ke : float
        kinetic energy
    pe : float
        potential energy
    """
    if ham.diagH1:
        ke = numpy.einsum("sii,sii->", ham.H1, G)
    else:
        ke = numpy.einsum("sij,sij->", ham.H1, G)

    Gkpq = numpy.zeros((2, len(ham.qvecs)), dtype=numpy.complex128)
    Gpmq = numpy.zeros((2, len(ham.qvecs)), dtype=numpy.complex128)
    Gprod = numpy.zeros((2, len(ham.qvecs)), dtype=numpy.complex128)

    ne = [system.nup, system.ndown]
    nq = numpy.shape(ham.qvecs)[0]

    for s in [0, 1]:
        # exchange_greens_function(nq, ham.ikpq_i, ham.ikpq_kpq, ham.ipmq_i,ham.ipmq_pmq, Gprod[s],G[s])
        # coulomb_greens_function(nq, ham.ikpq_i, ham.ikpq_kpq,  ham.ipmq_i, ham.ipmq_pmq,Gkpq[s], Gpmq[s],G[s])
        for iq in range(nq):
            Gkpq[s, iq], Gpmq[s, iq] = coulomb_greens_function_per_qvec(
                ham.ikpq_i[iq], ham.ikpq_kpq[iq], ham.ipmq_i[iq], ham.ipmq_pmq[iq], G[s]
            )
            Gprod[s, iq] = exchange_greens_function_per_qvec(
                ham.ikpq_i[iq], ham.ikpq_kpq[iq], ham.ipmq_i[iq], ham.ipmq_pmq[iq], G[s]
            )

    if two_rdm is None:
        two_rdm = numpy.zeros((2, 2, len(ham.qvecs)), dtype=numpy.complex128)

    two_rdm[0, 0] = numpy.multiply(Gkpq[0], Gpmq[0]) - Gprod[0]
    two_rdm[1, 1] = numpy.multiply(Gkpq[1], Gpmq[1]) - Gprod[1]
    two_rdm[0, 1] = numpy.multiply(Gkpq[0], Gpmq[1])
    two_rdm[1, 0] = numpy.multiply(Gkpq[1], Gpmq[0])

    essa = (1.0 / (2.0 * ham.vol)) * ham.vqvec.dot(two_rdm[0, 0])
    essb = (1.0 / (2.0 * ham.vol)) * ham.vqvec.dot(two_rdm[1, 1])
    eos = (1.0 / (2.0 * ham.vol)) * (
            ham.vqvec.dot(two_rdm[0, 1]) + ham.vqvec.dot(two_rdm[1, 0]))
    pe = essa + essb + eos

    if debug:
        ecoul, exx = 0., 0. 

        for s1 in range(2):
            exx -= (1.0 / (2.0 * ham.vol)) * ham.vqvec.dot(Gprod[s1])

            for s2 in range(2):
                ecoul += (1.0 / (2.0 * ham.vol)) * ham.vqvec.dot(numpy.multiply(Gkpq[s1], Gpmq[s2]))

        print(f'\n# ueg ecoul = {ecoul}')
        print(f'# ueg exx = {exx}')
        print(f'# ueg e2 = {(ecoul + exx)}')

    return (ke + pe, ke, pe)


# JHLFML
def build_J(system, Gpmq, Gkpq):
    J = [
        numpy.zeros((system.nbasis, system.nbasis), dtype=numpy.complex128),
        numpy.zeros((system.nbasis, system.nbasis), dtype=numpy.complex128),
    ]

    for iq, q in enumerate(system.qvecs):
        for idxi, i in enumerate(system.basis):
            for idxj, j in enumerate(system.basis):
                jpq = j + q
                idxjpq = system.lookup_basis(jpq)
                if (idxjpq is not None) and (idxjpq == idxi):
                    J[0][idxj, idxi] += (
                        (1.0 / (2.0 * system.vol)) * system.vqvec[iq] * (Gkpq[0][iq] + Gkpq[1][iq])
                    )

    for iq, q in enumerate(system.qvecs):
        for idxi, i in enumerate(system.basis):
            for idxj, j in enumerate(system.basis):
                jpq = j - q
                idxjmq = system.lookup_basis(jpq)
                if (idxjmq is not None) and (idxjmq == idxi):
                    J[0][idxj, idxi] += (
                        (1.0 / (2.0 * system.vol)) * system.vqvec[iq] * (Gpmq[0][iq] + Gpmq[1][iq])
                    )

    J[1] = J[0]

    return J


def build_K(system, G):
    K = numpy.zeros((2, system.nbasis, system.nbasis), dtype=numpy.complex128)
    for s in [0, 1]:
        for iq in range(len(system.vqvec)):
            for idxjmq, idxj in zip(system.ipmq_pmq[iq], system.ipmq_i[iq]):
                for idxkpq, idxk in zip(system.ikpq_kpq[iq], system.ikpq_i[iq]):
                    K[s, idxj, idxkpq] += (
                        -(1.0 / (2.0 * system.vol)) * system.vqvec[iq] * G[s][idxjmq, idxk]
                    )
        for iq in range(len(system.vqvec)):
            for idxjpq, idxj in zip(system.ikpq_kpq[iq], system.ikpq_i[iq]):
                for idxpmq, idxp in zip(system.ipmq_pmq[iq], system.ipmq_i[iq]):
                    K[s, idxj, idxpmq] += (
                        -(1.0 / (2.0 * system.vol)) * system.vqvec[iq] * G[s][idxjpq, idxp]
                    )
    return K


def fock_ueg(ham, G):
    """Fock matrix computation for uniform electron gas

    Parameters
    ----------
    ham : :class`ipie.legacy.hamiltonians.ueg`
        UEG hamiltonian class.
    G : :class:`numpy.ndarray`
        Green's function.
    Returns
    -------
    F : :class:`numpy.ndarray`
        Fock matrix (2, nbasis, nbasis).
    """
    nbsf = ham.nbasis
    nq = len(ham.qvecs)
    assert nq == len(ham.vqvec)

    Fock = numpy.zeros((2, nbsf, nbsf), dtype=numpy.complex128)
    Gkpq = numpy.zeros((2, nq), dtype=numpy.complex128)
    Gpmq = numpy.zeros((2, nq), dtype=numpy.complex128)

    for s in [0, 1]:
        coulomb_greens_function(
            nq,
            ham.ikpq_i,
            ham.ikpq_kpq,
            ham.ipmq_i,
            ham.ipmq_pmq,
            Gkpq[s],
            Gpmq[s],
            G[s],
        )

    J = build_J_opt(
        nq,
        ham.vqvec,
        ham.vol,
        ham.nbasis,
        ham.ikpq_i,
        ham.ikpq_kpq,
        ham.ipmq_i,
        ham.ipmq_pmq,
        Gkpq,
        Gpmq,
    )

    K = build_K_opt(
        nq,
        ham.vqvec,
        ham.vol,
        ham.nbasis,
        ham.ikpq_i,
        ham.ikpq_kpq,
        ham.ipmq_i,
        ham.ipmq_pmq,
        G,
    )

    for s in [0, 1]:
        Fock[s] = ham.H1[s] + J[s] + K[s]

    return Fock


def unit_test():
    import numpy as np

    from ipie.legacy.systems.ueg import UEG

    inputs = {"nup": 7, "ndown": 7, "rs": 1.0, "ecut": 2.0}
    system = UEG(inputs, True)
    system.build()
    nbsf = system.nbasis
    Pa = np.zeros([nbsf, nbsf], dtype=np.complex128)
    Pb = np.zeros([nbsf, nbsf], dtype=np.complex128)
    na = system.nup
    nb = system.ndown
    for i in range(na):
        Pa[i, i] = 1.0
    for i in range(nb):
        Pb[i, i] = 1.0
    P = np.array([Pa, Pb])
    etot, ekin, epot = local_energy_ueg(system, G=P)
    print(f"ERHF = {etot}, {ekin}, {epot}")

    from ipie.legacy.estimators.greens_function import gab
    from ipie.utils.linalg import exponentiate_matrix, reortho

    # numpy.random.seed()
    rCa = numpy.random.randn(nbsf, na)
    zCa = numpy.random.randn(nbsf, na)
    rCb = numpy.random.randn(nbsf, nb)
    zCb = numpy.random.randn(nbsf, nb)

    Ca = rCa + 1j * zCa
    Cb = rCb + 1j * zCb

    Ca, detR = reortho(Ca)
    Cb, detR = reortho(Cb)
    # S = print(Ca.dot(Cb.T))
    # print(S)
    # exit()
    Ca = numpy.array(Ca, dtype=numpy.complex128)
    Cb = numpy.array(Cb, dtype=numpy.complex128)
    P = [gab(Ca, Ca), gab(Cb, Cb)]
    # diff = P[0] - P[1]
    # print("fro = {}".format(numpy.linalg.norm(diff,ord='fro')))

    # solver = lib.diis.DIIS()

    # dt = 0.1
    # for i in range(100):
    #     # Compute Fock matrix
    #     Fock = fock_ueg(system, G=P)
    #     # Compute DIIS Errvec
    #     PFmFPa = P[0].dot(Fock[0]) - Fock[0].dot(P[0])
    #     PFmFPb = P[1].dot(Fock[1]) - Fock[1].dot(P[1])
    #     errvec = numpy.append(numpy.reshape(PFmFPa, nbsf*nbsf),numpy.reshape(PFmFPb, nbsf*nbsf))
    #     RMS = np.sqrt(np.dot(errvec, errvec))
    #     print ("{} {} {}".format(i,numpy.real(local_energy_ueg(system, P)), numpy.real(RMS)))
    #     # Form Fockvec
    #     Fock[0] = numpy.array(Fock[0])
    #     Fock[1] = numpy.array(Fock[1])
    #     Fockvec = numpy.append(numpy.reshape(Fock[0],nbsf*nbsf), numpy.reshape(Fock[1],nbsf*nbsf))
    #     # Extrapolate Fockvec
    #     # Fockvec = solver.update(Fockvec, xerr=errvec)

    #     # Apply Propagator
    #     Fock = numpy.reshape(Fockvec, (2, nbsf, nbsf))
    #     ea, Ca = numpy.linalg.eig(Fock[0])
    #     eb, Cb = numpy.linalg.eig(Fock[1])
    #     sort_perm = ea.argsort()
    #     ea.sort()
    #     Ca = Ca[:, sort_perm]
    #     sort_perm = eb.argsort()
    #     eb.sort()
    #     Cb = Cb[:, sort_perm]

    #     Ca = Ca[:,:na]
    #     Cb = Cb[:,:nb]
    #     Ca, detR = reortho(Ca)
    #     Cb, detR = reortho(Cb)

    #     P = [gab(Ca, Ca), gab(Cb, Cb)]
    #     # expF = [exponentiate_matrix(-dt*Fock[0]), exponentiate_matrix(-dt*Fock[1])]
    #     # Ca = expF[0].dot(Ca)
    #     # Cb = expF[1].dot(Cb)
    #     # diff = P[0] - P[1]
    #     # print("fro = {}".format(numpy.linalg.norm(diff,ord='fro')))


# -----------------------------------------------------------------------------
# System class.
# -----------------------------------------------------------------------------
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

    def __init__(self, options, verbose=False):
        if verbose:
            print("# Parsing input options.")

        self.name = "UEG"
        self.nup = options.get("nup")
        self.ndown = options.get("ndown")
        self.nelec = (self.nup, self.ndown)
        self.rs = options.get("rs")
        self.ecut = options.get("ecut")
        self.ktwist = numpy.array(options.get("ktwist", [0, 0, 0])).reshape(3)

        self.thermal = options.get("thermal", False)
        self._alt_convention = options.get("alt_convention", False)
        self.write_ints = options.get("write_integrals", False)
        
        self.sparse = True
        self.control_variate = False
        self.diagH1 = True

        # Total # of electrons.
        self.ne = self.nup + self.ndown
        # Spin polarisation.
        self.zeta = (self.nup - self.ndown) / self.ne
        # Density.
        self.rho = ((4.0 * numpy.pi) / 3.0 * self.rs**3.0) ** (-1.0)
        # Box Length.
        self.L = self.rs * (4.0 * self.ne * numpy.pi / 3.0) ** (1 / 3.0)
        # Volume
        self.vol = self.L**3.0
        # k-space grid spacing.
        self.kfac = 2 * numpy.pi / self.L
        # Fermi Wavevector (infinite system).
        self.kf = (3 * (self.zeta + 1) * numpy.pi**2 * self.ne / self.L**3) ** (1 / 3.0)
        # Fermi energy (inifinite systems).
        self.ef = 0.5 * self.kf**2
        # Core energy.
        self.ecore = 0.5 * self.ne * self.madelung()

        if verbose:
            if self.thermal:
                print("# Thermal UEG activated.")

            print(f"# Number of spin-up electrons: {self.nup:d}")
            print(f"# Number of spin-down electrons: {self.ndown:d}")
            print(f"# rs: {self.rs:6.4e}")
            print(f"# Spin polarisation (zeta): {self.zeta:6.4e}")
            print(f"# Electron density (rho): {self.rho:13.8e}")
            print(f"# Box Length (L): {self.L:13.8e}")
            print(f"# Volume: {self.vol:13.8e}")
            print(f"# k-space factor (2pi/L): {self.kfac:13.8e}")


    def build(self, verbose=False):
        # Get plane wave basis vectors and corresponding eigenvalues.
        self.sp_eigv, self.basis, self.nmax = self.sp_energies(
                                                self.ktwist, self.kfac, self.ecut)
        self.shifted_nmax = 2 * self.nmax
        self.imax_sq = numpy.dot(self.basis[-1], self.basis[-1])
        self.create_lookup_table()

        for i, k in enumerate(self.basis):
            assert i == self.lookup_basis(k)

        # Number of plane waves.
        self.nbasis = len(self.sp_eigv)
        self.nactive = self.nbasis
        self.ncore = 0
        self.nfv = 0
        self.mo_coeff = None
        
        # ---------------------------------------------------------------------
        T = numpy.diag(self.sp_eigv)
        h1e_mod = self.mod_one_body(T)
        self.H1 = numpy.array([T, T]) # Making alpha and beta.
        self.h1e_mod = numpy.array([h1e_mod, h1e_mod])

        # ---------------------------------------------------------------------
        # Allowed momentum transfers (4*ecut).
        eigs, qvecs, self.qnmax = self.sp_energies(self.ktwist, self.kfac, 4 * self.ecut)

        # Omit Q = 0 term.
        self.qvecs = numpy.copy(qvecs[1:])
        self.vqvec = numpy.array([self.vq(self.kfac * q) for q in self.qvecs])

        # Number of momentum transfer vectors / auxiliary fields.
        # Can reduce by symmetry but be stupid for the moment.
        self.nchol = len(self.qvecs)
        self.nfields = 2 * len(self.qvecs)
        self.get_momentum_transfers()

        if verbose:
            print(f"# Number of plane waves: {self.nbasis:d}")
            print(f"# Number of Cholesky vectors: {self.nchol:d}.")
            print(f"# Number of auxiliary fields: {self.nfields:d}.")
            print("# Constructing two-body potentials incore.")

        # ---------------------------------------------------------------------
        self.chol_vecs, self.iA, self.iB = self.two_body_potentials_incore()

        if self.write_ints:
            self.write_integrals()

        if verbose:
            print("# Approximate memory required for "
                  "two-body potentials: {:13.8e} GB.".format((3 * self.iA.nnz * 16 / (1024**3))))
            print("# Finished constructing two-body potentials.")
            print("# Finished building UEG object.")


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
        nmax = int(numpy.ceil(numpy.sqrt((2 * ecut))))

        spval = []
        vec = []
        kval = []

        for ni in range(-nmax, nmax + 1):
            for nj in range(-nmax, nmax + 1):
                for nk in range(-nmax, nmax + 1):
                    spe = 0.5 * (ni**2 + nj**2 + nk**2)

                    if spe <= ecut:
                        kijk = [ni, nj, nk]

                        # Reintroduce 2 \pi / L factor.
                        ek = 0.5 * numpy.dot(numpy.array(kijk) + ks, numpy.array(kijk) + ks)
                        kval.append(kijk)
                        spval.append(kfac**2 * ek)

        # Sort the arrays in terms of increasing energy.
        spval = numpy.array(spval)
        ix = numpy.argsort(spval, kind="mergesort")
        spval = spval[ix]
        kval = numpy.array(kval)[ix]
        return spval, kval, nmax


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
        return ((k[0] + self.nmax)
                + self.shifted_nmax * (k[1] + self.nmax)
                + self.shifted_nmax * self.shifted_nmax * (k[2] + self.nmax))


    def get_momentum_transfers(self):
        """Get arrays of plane wave basis vectors connected by momentum transfers Q."""
        nlimit = self.nup
        if self.thermal:
            nlimit = self.nbasis

        self.ikpq_i = []
        self.ikpq_kpq = []

        for iq, q in enumerate(self.qvecs):
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
        
        # ---------------------------------------------------------------------
        self.ipmq_i = []
        self.ipmq_pmq = []

        for iq, q in enumerate(self.qvecs):
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

        for iq, q in enumerate(self.qvecs):
            self.ikpq_i[iq] = numpy.array(self.ikpq_i[iq], dtype=numpy.int64)
            self.ikpq_kpq[iq] = numpy.array(self.ikpq_kpq[iq], dtype=numpy.int64)
            self.ipmq_i[iq] = numpy.array(self.ipmq_i[iq], dtype=numpy.int64)
            self.ipmq_pmq[iq] = numpy.array(self.ipmq_pmq[iq], dtype=numpy.int64)


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
        c2 = (3.0 / (4.0 * numpy.pi)) ** (1.0 / 3.0)
        return c1 * c2 / (self.ne ** (1.0 / 3.0) * self.rs)

    
    def mod_one_body(self, T):
        """Absorb the diagonal term of the two-body Hamiltonian to the one-body term.
        Essentially adding the third term in Eq.(11b) of Phys. Rev. B 75, 245123.

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

        fac = 1.0 / (2.0 * self.vol)
        for i, ki in enumerate(self.basis):
            for j, kj in enumerate(self.basis):
                if i != j:
                    q = self.kfac * (ki - kj)
                    h1e_mod[i, i] = h1e_mod[i, i] - fac * self.vq(q)

        return h1e_mod


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
        return 4 * numpy.pi / numpy.dot(q, q)


    def density_operator(self, iq):
        """Density operator as defined in Eq.(6) of Phys. Rev. B 75, 245123.

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
            dtype=numpy.complex128)
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

        for iq, q in enumerate(self.qvecs):
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

        for iq, q in enumerate(self.qvecs):
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
                piovol = numpy.pi / (self.vol)
                factor = (piovol / numpy.dot(qscaled, qscaled)) ** 0.5

                for innz, kpq in enumerate(rho_ikpq_kpq[iq]):
                    row_index += [rho_ikpq_kpq[iq][innz] + rho_ikpq_i[iq][innz] * self.nbasis]
                    col_index += [iq]
                    values += [factor]

        else:
            for iq in range(nq):
                qscaled = self.kfac * self.qvecs[iq]
                # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
                piovol = numpy.pi / (self.vol)
                factor = (piovol / numpy.dot(qscaled, qscaled)) ** 0.5

                for innz, kpq in enumerate(rho_ikpq_kpq[iq]):
                    row_index += [rho_ikpq_kpq[iq][innz] * self.nbasis + rho_ikpq_i[iq][innz]]
                    col_index += [iq]
                    values += [factor]

        rho_q = scipy.sparse.csc_matrix(
            (values, (row_index, col_index)),
            shape=(self.nbasis * self.nbasis, nq),
            dtype=numpy.complex128)
        return rho_q


    def two_body_potentials_incore(self):
        """Calculate A and B of Eq.(13) of PRB(75)245123 for a given plane-wave vector q

        Returns
        -------
        iA : numpy array
            Eq.(13a)

        iB : numpy array
            Eq.(13b)
        """
        rho_q = self.scaled_density_operator_incore(False)
        rho_qH = self.scaled_density_operator_incore(True)
        iA = 1j * (rho_q + rho_qH)
        iB = -(rho_q - rho_qH)
        return (rho_q, iA, iB)


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
            eri_chol.toarray().reshape((self.nbasis, self.nbasis, self.nbasis, self.nbasis)).real)
        eri_chol = eri_chol.transpose(0, 1, 3, 2)
        return eri_chol


    def eri_8(self):
        """Compute 8-fold symmetric integrals. Useful for running standard 
        quantum chemistry methods,"""
        eri = self.eri_4()
        U = self.compute_real_transformation()
        eri0 = numpy.einsum("mp,mnls->pnls", U.conj(), eri, optimize=True)
        eri1 = numpy.einsum("nq,pnls->pqls", U, eri0, optimize=True)
        eri2 = numpy.einsum("lr,pqls->pqrs", U.conj(), eri1, optimize=True)
        eri3 = numpy.einsum("st,pqrs->pqrt", U, eri2, optimize=True).real
        return eri3
    

    def write_integrals(self, filename="ueg_integrals.h5"):
        write_qmcpack_sparse(
            self.H1[0],
            2 * self.chol_vecs.toarray(),
            self.nelec,
            self.nbasis,
            enuc=0.0,
            filename=filename)


if __name__ == "__main__":
    unit_test()
