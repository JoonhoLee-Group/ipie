import numpy

try:
    import mpi4py

    mpi4py.rc.recv_mprobe = False
    from mpi4py import MPI

    mpi_sum = MPI.SUM
except ImportError:
    mpi_sum = None
import scipy.linalg

from ipie.estimators.utils import H5EstimatorHelper
from ipie.legacy.estimators.greens_function import gab, gab_multi_ghf
from ipie.legacy.propagation.generic import (
    back_propagate_generic, construct_propagator_matrix_generic)
from ipie.legacy.propagation.hubbard import (back_propagate_single,
                                             back_propagate_single_ghf,
                                             construct_propagator_matrix,
                                             construct_propagator_matrix_ghf)
from ipie.legacy.propagation.operations import propagate_single
from ipie.utils.linalg import reortho


class ITCF(object):
    """Class for computing ITCF estimates.

    Parameters
    ----------
    itcf : dict
        Input options for ITCF estimates :

            - tmax : float
                Maximum value of imaginary time to calculate ITCF to.
            - stable : bool
                If True use the stabalised algorithm of Feldbacher and Assad.
            - mode : string / list
                How much of the ITCF to save to file:
                    'full' : print full ITCF.
                    'diagonal' : print diagonal elements of ITCF.
                    elements : list : print select elements defined from list.
            - kspace : bool
                If True evaluate correlation functions in momentum space.

    root : bool
        True if on root/master processor.
    h5f : :class:`h5py.File`
        Output file object.
    qmc : :class:`ipie.state.QMCOpts` object.
        Container for qmc input options.
    system : system object
        System object.
    trial : :class:`ipie.trial_wavefunction.X' object
        Trial wavefunction class.
    dtype : complex or float
        Output type.
    BT2 : :class:`numpy.ndarray`
        One-body propagator for back propagation.

    Attributes
    ----------
    nmax : int
        Number of back propagation steps to perform.
    header : list
        Header sfor back propagated estimators.
    key : dict
        Explanation of spgf data structure.
    spgf : :class:`numpy.ndarray`
        Storage for single-particle greens function (SPGF).
    spgf_global : :class:`numpy.ndarray`
        Store for ITCF accross all processors.
    rspace_unit : :class:`ipie.estimators.H5EstimatorHelper`
        Output dataset for real space itcfs.
    kspace_unit : :class:`ipie.estimators.H5EstimatorHelper`
        Output dataset for real space itcfs.
    """

    def __init__(self, itcf, qmc, trial, root, h5f, system, dtype, BT2):
        self.stable = itcf.get("stable", True)
        self.restore_weights = itcf.get("restore_weights", True)
        self.tmax = itcf.get("tau_max", 0.0)
        self.teqlb = itcf.get("tau_eqlb", 0.0)
        self.mode = itcf.get("mode", "full")
        self.stack_size = itcf.get("stack_size", 1)
        self.nmax = int(self.tmax / qmc.dt)
        self.dt = qmc.dt
        self.ntau = int(self.nmax / self.stack_size)
        self.neqlb = int(self.teqlb / qmc.dt)
        self.nprop_tot = self.nmax + self.neqlb
        self.nstblz = qmc.nstblz
        self.denom = 0
        self.BT2 = BT2
        self.kspace = itcf.get("kspace", False)
        # self.spgf(i,j,k,l,m) gives the (l,m)th element of the spin-j(=0 for up
        # and 1 for down) k-ordered(0=greater,1=lesser) imaginary time green's
        # function at time i.
        # +1 in the first dimension is for the green's function at time tau = 0.
        nbasis = system.nbasis
        self.spgf_shape = (self.ntau + 1, 2, 2, nbasis, nbasis)
        self.spgf = numpy.zeros(shape=self.spgf_shape, dtype=trial.G.dtype)
        self.global_array = numpy.zeros(
            shape=(1 + numpy.prod(self.spgf_shape),), dtype=trial.G.dtype
        )
        # self.global_array = numpy.zeros(shape=(self.nmax+1,2,2,nbasis,nbasis),
        # dtype=trial.G.dtype)
        if trial.type == "GHF":
            self.I = numpy.identity(trial.psi.shape[1], dtype=trial.psi.dtype)
            self.initial_greens_function = self.initial_greens_function_ghf
            self.accumulate = self.accumulate_ghf
            self.back_propagate_single = back_propagate_single_ghf
            self.construct_propagator_matrix = construct_propagator_matrix_ghf
            if self.stable:
                self.increment_tau = self.increment_tau_ghf_stable
            else:
                self.increment_tau = self.increment_tau_ghf_unstable
        else:
            self.I = numpy.identity(trial.psi.shape[0], dtype=trial.psi.dtype)
            self.initial_greens_function = self.initial_greens_function_uhf
            self.accumulate = self.accumulate_uhf
            if system.name == "Generic":
                self.back_propagate_single = back_propagate_generic
                self.construct_propagator_matrix = construct_propagator_matrix_generic
            if system.name == "Hubbard":
                self.back_propagate_single = back_propagate_single
                self.construct_propagator_matrix = construct_propagator_matrix
            if self.stable:
                self.increment_tau = self.increment_tau_uhf_stable
            else:
                self.increment_tau = self.increment_tau_uhf_unstable
        if self.stable:
            self.calculate_spgf = self.calculate_spgf_stable
        else:
            self.calculate_spgf = self.calculate_spgf_unstable
        self.keys = [["up", "down"], ["greater", "lesser"]]
        # I don't like list indexing so stick with numpy.
        if root:
            if self.mode == "full":
                shape = (qmc.nsteps // (self.nmax),) + self.spgf.shape
            elif self.mode == "diagonal":
                shape = (qmc.nsteps // (self.nmax), self.nmax + 1, 2, 2, nbasis)
            else:
                shape = (qmc.nsteps // (self.nmax), self.nmax + 1, 2, 2, len(self.mode))
            spgfs = h5f.create_group("single_particle_greens_function")
            self.rspace_unit = H5EstimatorHelper(
                spgfs, "real_space", shape, self.spgf.dtype
            )
            if self.kspace:
                self.kspace_unit = H5EstimatorHelper(
                    spgfs, "k_space", shape, self.spgf.dtype
                )

    def update(self, system, qmc, trial, psi, step, free_projection=False):
        """Update estimators

        Parameters
        ----------
        system : system object in general.
            Container for model input options.
        qmc : :class:`ipie.state.QMCOpts` object.
            Container for qmc input options.
        trial : :class:`ipie.trial_wavefunction.X' object
            Trial wavefunction class.
        psi : :class:`ipie.legacy.walkers.Walkers` object
            CPMC wavefunction.
        step : int
            Current simulation step
        free_projection : bool
            True if doing free projection.
        """
        if step % self.nprop_tot == 0:
            self.calculate_spgf(system, psi, trial)

    def calculate_spgf_unstable(self, system, psi, trial):
        r"""Calculate imaginary time single-particle green's function.

        This uses the naive unstable algorithm.

        On return the spgf estimator array will have been updated.

        Parameters
        ----------
        system : system object in general.
            Container for model input options.
        psi : :class:`ipie.legacy.walkers.Walkers` object
            CPMC wavefunction.
        trial : :class:`ipie.trial_wavefunction.X' object
            Trial wavefunction class.
        """

        nup = system.nup
        M = system.nbasis
        if self.restore_weights:
            self.denom = sum(
                w.weight * w.stack.wfac[0] / w.stack.wfac[1] for w in psi.walkers
            )
        else:
            self.denom = sum(w.weight for w in psi.walkers)
        for ix, w in enumerate(psi.walkers):
            # 1. Construct psi_left for first step in algorithm by back
            # propagating the input back propagated left hand wfn.
            # Note we use the first nmax fields for estimating the ITCF.
            # configs = w.field_configs.get_superblock()[0]
            phi_left = trial.psi.copy()
            self.back_propagate_single(
                phi_left, w.stack, system, self.nstblz, self.BT2, self.dt
            )
            (Ggr, Gls) = self.initial_greens_function(
                phi_left, w.phi_right, trial, nup, w.weights
            )
            # 2. Calculate G(n,n). This is the equal time Green's function at
            # the step where we began saving auxilary fields (constructed with
            # psi_left back propagated along this path.)
            if self.restore_weights:
                wfac = w.weight * w.stack.wfac[0] / w.stack.wfac[1]
            else:
                wfac = w.weight
            self.accumulate(0, wfac, Ggr, Gls, M)
            # 3. Construct ITCF by moving forwards in imaginary time from time
            # slice n along our auxiliary field path.
            for ic in range(self.nmax // w.stack.stack_size):
                # B takes the state from time n to time n+1.
                B = w.stack.get(it)
                (Ggr, Gls) = self.increment_tau(Ggr, Gls, B)
                self.accumulate(ic + 1, wfac, Ggr, Gls, M)
            w.stack.reset()
        # copy current walker distribution to initial (right hand) wavefunction
        # for next estimate of ITCF
        psi.copy_init_wfn()

    def calculate_spgf_stable(self, system, psi, trial):
        """Calculate imaginary time single-particle green's function.

        This uses the stable algorithm as outlined in:
        Feldbacher and Assad, Phys. Rev. B 63, 073105.

        On return the spgf estimator array will have been updated.

        Parameters
        ----------
        system : system object in general.
            Container for model input options.
        psi : :class:`ipie.legacy.walkers.Walkers` object
            CPMC wavefunction.
        trial : :class:`ipie.trial_wavefunction.X' object
            Trial wavefunction class.
        """

        nup = system.nup
        M = system.nbasis
        if self.restore_weights:
            self.denom = sum(
                w.weight * w.stack.wfac[0] / w.stack.wfac[1] for w in psi.walkers
            )
        else:
            self.denom = sum(w.weight for w in psi.walkers)
        for ix, w in enumerate(psi.walkers):
            Ggr = numpy.array([self.I.copy(), self.I.copy()])
            Gls = numpy.array([self.I.copy(), self.I.copy()])
            # 1. Construct psi_L for first step in algorithm by back
            # propagating the input back propagated left hand wfn.
            # Note we use the first itcf_nmax fields for estimating the ITCF.
            # We store for intermediate back propagated left-hand wavefunctions.
            # This leads to more stable equal time green's functions compared to
            # that found by multiplying psi_L^n by B^{-1}(x^(n)) factors.
            phi_left = trial.psi.copy()
            psi_Ls = self.back_propagate_single(
                phi_left, w.stack, system, self.nstblz, self.BT2, self.dt, store=True
            )
            # 2. Calculate G(n,n). This is the equal time Green's function at
            # the step where we began saving auxilary fields (constructed with
            # psi_L back propagated along this path.)
            (Ggr_nn, Gls_nn) = self.initial_greens_function(
                phi_left, w.phi_right, trial, nup, w.weights
            )
            # 3. Construct ITCF by moving forwards in imaginary time from time
            # slice n along our auxiliary field path.
            if self.restore_weights:
                wfac = w.weight * w.stack.wfac[0] / w.stack.wfac[1]
            else:
                wfac = w.weight
            self.accumulate(0, wfac, Ggr_nn, Gls_nn, M)
            for ic in range(self.nmax // w.stack.stack_size):
                # B takes the state from time n to time n+1.
                B = w.stack.get(ic)
                # G is the cumulative product of stabilised short-time ITCFs.
                # The first term in brackets is the G(n+1,n) which should be
                # well conditioned.
                (Ggr, Gls) = self.increment_tau(Ggr, Gls, B, Ggr_nn, Gls_nn)
                self.accumulate(ic + 1, wfac, Ggr, Gls, M)
                # Construct equal-time green's function shifted forwards along
                # the imaginary time interval. We need to update |psi_L> =
                # (B(c)^{dagger})^{-1}|psi_L> and |psi_R> = B(c)|psi_R>, where c
                # is the current configution in this loop. Note that we store
                # |psi_L> along the path, so we don't need to remove the
                # propagator matrices.
                L = psi_Ls[len(psi_Ls) - ic - 1]
                propagate_single(w.phi_right, system, B)
                if ic != 0 and ic % self.nstblz == 0:
                    (w.phi_right[:, :nup], R) = reortho(w.phi_right[:, :nup])
                    (w.phi_right[:, nup:], R) = reortho(w.phi_right[:, nup:])
                (Ggr_nn, Gls_nn) = self.initial_greens_function(
                    L, w.phi_right, trial, nup, w.weights
                )
            w.stack.reset()
        # copy current walker distribution to initial (right hand) wavefunction
        # for next estimate of ITCF
        psi.copy_init_wfn()

    def initial_greens_function_uhf(self, A, B, trial, nup, weights):
        r"""Compute initial green's function at timestep n for UHF wavefunction.

        Here we actually compute the equal-time green's function:

        .. math::

            G_{ij} = \langle c_i c_j^{\dagger} \rangle

        Parameters
        ----------
        A : :class:`numpy.ndarray`
            Left hand wavefunction for green's function.
        B : :class:`numpy.ndarray`
            Left hand wavefunction for green's function.
        trial : :class:`ipie.trial_wavefunction.X' object
            Trial wavefunction class.
        nup : int
            Number of up electrons.
        weight : :class:`numpy.ndarray`
            Any GS orthogonalisation factors which need to be included.

        Returns
        -------
        G_nn : :class:`numpy.ndarray`
            Green's function.
        """
        Ggr_up = self.I - gab(A[:, :nup], B[:, :nup])
        Ggr_down = self.I - gab(A[:, nup:], B[:, nup:])
        Gls_up = self.I - Ggr_up
        Gls_down = self.I - Ggr_down
        return (numpy.array([Ggr_up, Ggr_down]), numpy.array([Gls_up, Gls_down]))

    def initial_greens_function_ghf(self, A, B, trial, nup, weights):
        r"""Compute initial green's function at timestep n for GHF wavefunction.

        Here we actually compute the equal-time green's function:

        .. math::

            G_{ij} = \langle c_i c_j^{\dagger} \rangle

        Parameters
        ----------
        A : :class:`numpy.ndarray`
            Left hand wavefunction for green's function.
        B : :class:`numpy.ndarray`
            Left hand wavefunction for green's function.
        trial : :class:`ipie.trial_wavefunction.X' object
            Trial wavefunction class.
        nup : int
            Number of up electrons.
        weight : :class:`numpy.ndarray`
            Any GS orthogonalisation factors which need to be included.

        Returns
        -------
        G_nn : :class:`numpy.ndarray`
            Green's function.
        """
        GAB = gab_multi_ghf(A, B, trial.coeffs, weights)
        Ggr = self.I - GAB
        Gls = self.I - Ggr
        return (Ggr, Gls)

    def accumulate_uhf(self, idx, weight, Ggr, Gls, nbasis):
        """Accumulate ITCF for UHF wavefunction.

        Parameters
        ----------
        idx : int
            Time index.
        weight : float
            Walker weight.
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Gls : :class:`numpy.ndarray`
            Lesser ITCF.
        nbasis : int
            Number of basis functions.
        """
        self.spgf[idx, 0, 0] += weight * Ggr[0].real
        self.spgf[idx, 1, 0] += weight * Ggr[1].real
        self.spgf[idx, 0, 1] += weight * Gls[0].real
        self.spgf[idx, 1, 1] += weight * Gls[1].real

    def accumulate_ghf(self, idx, weight, Ggr, Gls, nbasis):
        """Accumulate ITCF for GHF wavefunction.

        Parameters
        ----------
        idx : int
            Time index.
        weight : float
            Walker weight.
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Gls : :class:`numpy.ndarray`
            Lesser ITCF.
        nbasis : int
            Number of basis functions.
        """
        self.spgf[idx, 0, 0] += weight * Ggr[:nbasis, :nbasis].real
        self.spgf[idx, 1, 0] += weight * Ggr[nbasis:, nbasis:].real
        self.spgf[idx, 0, 1] += weight * Gls[:nbasis, :nbasis].real
        self.spgf[idx, 1, 1] += weight * Gls[nbasis:, nbasis:].real

    def increment_tau_ghf_unstable(self, Ggr, Gls, B, Gnn_gr=None, Gnn_ls=None):
        """Update ITCF to next time slice. Unstable algorithm, GHF format.

        Parameters
        ----------
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Lesser ITCF.
        Gls : :class:`numpy.ndarray`
            Propagator matrix.
        G_nn_gr : :class:`numpy.ndarray`, not used
            Greater equal-time green's function.
        G_nn_ls : :class:`numpy.ndarray`, not used
            Lesser equal-time green's function.

        Returns
        -------
        Ggr : :class:`numpy.ndarray`
            Updated greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Updated lesser ITCF.
        """
        Ggr = B.dot(Ggr)
        Gls = Gls.dot(scipy.linalg.inv(B))
        return Ggr, Gls

    def increment_tau_uhf_unstable(self, Ggr, Gls, B, Gnn_gr=None, Gnn_ls=None):
        """Update ITCF to next time slice. Unstable algorithm, UHF format.

        Parameters
        ----------
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Lesser ITCF.
        Gls : :class:`numpy.ndarray`
            Propagator matrix.
        G_nn_gr : :class:`numpy.ndarray`, not used
            Greater equal-time green's function.
        G_nn_ls : :class:`numpy.ndarray`, not used
            Lesser equal-time green's function.

        Returns
        -------
        Ggr : :class:`numpy.ndarray`
            Updated greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Updated lesser ITCF.
        """
        Ggr[0] = B[0].dot(Ggr[0])
        Ggr[1] = B[1].dot(Ggr[1])
        Gls[0] = Gls[0].dot(scipy.linalg.inv(B[0]))
        Gls[1] = Gls[1].dot(scipy.linalg.inv(B[1]))
        return Ggr, Gls

    def increment_tau_uhf_stable(self, Ggr, Gls, B, Gnn_gr, Gnn_ls):
        """Update ITCF to next time slice. Stable algorithm, UHF format.

        Parameters
        ----------
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Lesser ITCF.
        Gls : :class:`numpy.ndarray`
            Propagator matrix.
        G_nn_gr : :class:`numpy.ndarray`
            Greater equal-time green's function.
        G_nn_ls : :class:`numpy.ndarray`
            Lesser equal-time green's function.

        Returns
        -------
        Ggr : :class:`numpy.ndarray`
            Updated greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Updated lesser ITCF.
        """
        Ggr[0] = (B[0].dot(Gnn_gr[0])).dot(Ggr[0])
        Ggr[1] = (B[1].dot(Gnn_gr[1])).dot(Ggr[1])
        Gls[0] = Gls[0].dot(Gnn_ls[0].dot(scipy.linalg.inv(B[0])))
        Gls[1] = Gls[0].dot(Gnn_ls[1].dot(scipy.linalg.inv(B[1])))
        return Ggr, Gls

    def increment_tau_ghf_stable(self, Ggr, Gls, B, Gnn_gr, Gnn_ls):
        """Update ITCF to next time slice. Stable algorithm, GHF format.

        Parameters
        ----------
        Ggr : :class:`numpy.ndarray`
            Greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Lesser ITCF.
        Gls : :class:`numpy.ndarray`
            Propagator matrix.
        G_nn_gr : :class:`numpy.ndarray`
            Greater equal-time green's function.
        G_nn_ls : :class:`numpy.ndarray`
            Lesser equal-time green's function.

        Returns
        -------
        Ggr : :class:`numpy.ndarray`
            Updated greater ITCF.
        Ggr : :class:`numpy.ndarray`
            Updated lesser ITCF.
        """
        Ggr = (B.dot(Gnn_gr)).dot(Ggr)
        Gls = (Gnn_ls.dot(scipy.linalg.inv(B))).dot(Gls)
        return Ggr, Gls

    def print_step(self, comm, nprocs, step, nmeasure=1, free_projection=False):
        """Print ITCF to file.

        Parameters
        ----------
        comm :
            MPI communicator.
        nprocs : int
            Number of processors.
        step : int
            Current iteration number.
        nmeasure : int
            Number of steps between measurements.
        """
        if step != 0 and step % self.nprop_tot == 0:
            sendbuf = numpy.concatenate([numpy.array([self.denom]), self.spgf.ravel()])
            comm.Reduce(sendbuf, self.global_array, op=mpi_sum)
            if comm.Get_rank() == 0:
                itcf = self.global_array[1:].reshape(self.spgf_shape)
                # print(self.global_array[0], self.global_array[1:].reshape(self.spgf_shape)[0,0,0].diagonal())
                denom = self.global_array[0]
                self.to_file(self.rspace_unit, itcf / denom)
                # if self.kspace:
                # M = self.spgf.shape[-1]
                # # FFT the real space Green's function.
                # # Todo : could just use numpy.fft.fft....
                # # spgf_k = numpy.einsum('ik,rqpkl,lj->rqpij', self.P,
                # # spgf, self.P.conj().T) / M
                # spgf_k = numpy.fft.fft2(self.spgf_global)
                # if self.spgf.dtype == complex:
                # self.to_file(self.kspace_unit, spgf_k/nprocs)
                # else:
                # self.to_file(self.kspace_unit, spgf_k.real/nprocs)
            self.zero()

    def to_file(self, group, spgf):
        """Push ITCF to hdf5 group.

        Parameters
        ----------
        group: string
            HDF5 group name.
        spgf : :class:`numpy.ndarray`
            Single-particle Green's function (SPGF).
        """
        if self.mode == "full":
            group.push(spgf)
        elif self.mode == "diagonal":
            group.push(spgf.diagonal(axis1=3, axis2=4))
        else:
            group.push(numpy.array([g[mode] for g in spgf]))

    def zero(self):
        """Zero (in the appropriate sense) various estimator arrays."""
        self.spgf[:] = 0
        self.denom = 0
        self.global_array[:] = 0
