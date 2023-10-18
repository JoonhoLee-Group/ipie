import numpy
import scipy.linalg

from ipie.thermal.estimators.thermal import one_rdm_from_G, particle_number
from ipie.thermal.walkers.stack import PropagatorStack
from ipie.utils.misc import get_numeric_names, update_stack
from ipie.walkers.base_walkers import BaseWalkers
from ipie.thermal.trial.one_body import OneBody

class UHFThermalWalkers(BaseWalkers):
    def __init__(
        self,
        trial: OneBody,
        nbasis: int,
        nwalkers: int,
        nstack = None,
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
        self.nstack = nstack

        if self.nstack == None:
            self.nstack = trial.nstack

        if (self.nslice // self.nstack) * self.nstack != self.nslice:
            if verbose:
                print("# Input stack size does not divide number of slices.")
            self.nstack = update_stack(self.nstack, self.nslice, verbose)

        if self.nstack > trial.nstack:
            if verbose:
                print("# Walker stack size differs from that estimated from " "Trial density matrix.")
                print(f"# Be careful. cond(BT)**nstack: {trial.cond ** self.nstack:10.3e}.")

        self.stack_length = self.nslice // self.nstack
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
            print(f"# Walker stack size: {self.nstack}")
            print(f"# Using low rank trick: {self.lowrank}")

        self.stack = [PropagatorStack(
            self.nstack,
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
            self.greens_function_qr_strat(iw)
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

        self.buff_names, self.buff_size = get_numeric_names(self.__dict__)


    def greens_function(self, iw, slice_ix=None, inplace=True):
        """Return the Green's function for walker `iw`.
        """
        if self.lowrank:
            return self.stack[iw].G # G[0] = Ga, G[1] = Gb
        else:
            return self.greens_function_qr_strat(iw, slice_ix=slice_ix, inplace=inplace)


    def greens_function_qr_strat(self, iw, slice_ix=None, inplace=True):
        """Compute the Green's function for walker with index `iw` at time 
        `slice_ix`. Uses the Stratification method (DOI 10.1109/IPDPS.2012.37)
        """
        stack_iw = self.stack[iw]

        if slice_ix == None:
            slice_ix = stack_iw.time_slice

        bin_ix = slice_ix // stack_iw.nstack
        # For final time slice want first block to be the rightmost (for energy
        # evaluation).
        if bin_ix == stack_iw.nbins:
            bin_ix = -1

        Ga_iw, Gb_iw = None, None
        if not inplace:
            Ga_iw = numpy.zeros(self.Ga[iw].shape, self.Ga.dtype)
            Gb_iw = numpy.zeros(self.Gb[iw].shape, self.Gb.dtype)

        for spin in [0, 1]:
            # Need to construct the product A(l) = B_l B_{l-1}..B_L...B_{l+1} in
            # stable way. Iteratively construct column pivoted QR decompositions
            # (A = QDT) starting from the rightmost (product of) propagator(s).
            B = stack_iw.get((bin_ix + 1) % stack_iw.nbins)

            (Q1, R1, P1) = scipy.linalg.qr(B[spin], pivoting=True, check_finite=False)
            # Form D matrices
            D1 = numpy.diag(R1.diagonal())
            D1inv = numpy.diag(1.0 / R1.diagonal())
            T1 = numpy.einsum("ii,ij->ij", D1inv, R1)
            # permute them
            T1[:, P1] = T1[:, range(self.nbasis)]

            for i in range(2, stack_iw.nbins + 1):
                ix = (bin_ix + i) % stack_iw.nbins
                B = stack_iw.get(ix)
                C2 = numpy.dot(numpy.dot(B[spin], Q1), D1)
                (Q1, R1, P1) = scipy.linalg.qr(C2, pivoting=True, check_finite=False)
                # Compute D matrices
                D1inv = numpy.diag(1.0 / R1.diagonal())
                D1 = numpy.diag(R1.diagonal())
                tmp = numpy.einsum("ii,ij->ij", D1inv, R1)
                tmp[:, P1] = tmp[:, range(self.nbasis)]
                T1 = numpy.dot(tmp, T1)

            # G^{-1} = 1+A = 1+QDT = Q (Q^{-1}T^{-1}+D) T
            # Write D = Db^{-1} Ds
            # Then G^{-1} = Q Db^{-1}(Db Q^{-1}T^{-1}+Ds) T
            Db = numpy.zeros(B[spin].shape, B[spin].dtype)
            Ds = numpy.zeros(B[spin].shape, B[spin].dtype)
            for i in range(Db.shape[0]):
                absDlcr = abs(Db[i, i])
                if absDlcr > 1.0:
                    Db[i, i] = 1.0 / absDlcr
                    Ds[i, i] = numpy.sign(D1[i, i])
                else:
                    Db[i, i] = 1.0
                    Ds[i, i] = D1[i, i]

            T1inv = scipy.linalg.inv(T1, check_finite=False)
            # C = (Db Q^{-1}T^{-1}+Ds)
            C = numpy.dot(numpy.einsum("ii,ij->ij", Db, Q1.conj().T), T1inv) + Ds
            Cinv = scipy.linalg.inv(C, check_finite=False)

            # Then G = T^{-1} C^{-1} Db Q^{-1}
            # Q is unitary.
            if inplace:
                if spin == 0:
                    self.Ga[iw] = numpy.dot(
                        numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T))
                else:
                    self.Gb[iw] = numpy.dot(
                        numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T))

            else:
                if spin == 0:
                    Ga_iw = numpy.dot(
                        numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T))

                else:
                    Gb_iw = numpy.dot(
                        numpy.dot(T1inv, Cinv), numpy.einsum("ii,ij->ij", Db, Q1.conj().T))

        return Ga_iw, Gb_iw

    
    # For compatibiltiy with BaseWalkers class.
    def reortho(self):
        pass

    def reortho_batched(self):
        pass
