import cmath
import math
import sys
import time

import numpy
import scipy.sparse.linalg
from scipy.linalg import sqrtm

from ipie.legacy.estimators.thermal import (inverse_greens_function_qr,
                                            one_rdm_from_G)
from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.propagation.operations import kinetic_real
from ipie.utils.linalg import exponentiate_matrix


class PlaneWave(object):
    """PlaneWave class"""

    def __init__(
        self, system, hamiltonian, trial, qmc, options={}, verbose=False, lowrank=False
    ):
        self.verbose = verbose
        if verbose:
            print("# Parsing plane wave propagator input options.")
        # Input options
        self.hs_type = "plane_wave"
        self.free_projection = options.get("free_projection", False)
        self.optimised = options.get("optimised", True)
        self.lowrank = lowrank
        self.exp_nmax = options.get("expansion_order", 6)
        self.nstblz = qmc.nstblz
        self.fb_bound = options.get("fb_bound", 1.0)
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j * self.sqrt_dt
        self.num_vplus = hamiltonian.nfields // 2
        self.mf_shift = self.construct_mf_shift(hamiltonian, trial)
        if verbose:
            print(
                "# Absolute value of maximum component of mean field shift: "
                "{:13.8e}.".format(numpy.max(numpy.abs(self.mf_shift)))
            )
        if verbose:
            print("# Number of fields = %i" % hamiltonian.nfields)
            print("# Using lowrank propagation: {}".format(self.lowrank))

        self.vbias = numpy.zeros(hamiltonian.nfields, dtype=numpy.complex128)

        # Constant core contribution modified by mean field shift.
        mf_core = hamiltonian.ecore

        self.construct_one_body_propagator(hamiltonian, qmc.dt)

        self.BT = trial.dmat
        self.BTinv = trial.dmat_inv

        self.mf_const_fac = 1

        # todo : ?
        self.BT_BP = self.BT
        self.nstblz = qmc.nstblz

        self.ebound = (2.0 / self.dt) ** 0.5
        self.mean_local_energy = 0

        # self.propagate_walker_phaseless = self.propagate_walker_phaseless_full_rank
        if self.lowrank:
            self.propagate_walker_free = self.propagate_walker_phaseless_low_rank
            self.propagate_walker_phaseless = self.propagate_walker_phaseless_low_rank
        else:
            self.propagate_walker_free = self.propagate_walker_free_full_rank
            self.propagate_walker_phaseless = self.propagate_walker_phaseless_full_rank

        if self.free_projection:
            if verbose:
                print("# Using free projection")
            self.propagate_walker = self.propagate_walker_free
        else:
            if verbose:
                print("# Using phaseless approximation")
            self.propagate_walker = self.propagate_walker_phaseless
        if verbose:
            print("# Finished setting up propagator.")
        self.nfb_trig = False

    def construct_mf_shift(self, hamiltonian, trial):
        P = one_rdm_from_G(trial.G)
        P = P.reshape(2, hamiltonian.nbasis * hamiltonian.nbasis)
        mf_shift = numpy.zeros(hamiltonian.nfields, numpy.complex128)
        mf_shift[: self.num_vplus] = P[0].T * hamiltonian.iA + P[1].T * hamiltonian.iA
        mf_shift[self.num_vplus :] = P[0].T * hamiltonian.iB + P[1].T * hamiltonian.iB
        return mf_shift

    def construct_one_body_propagator(self, hamiltonian, dt):
        """Construct the one-body propagator Exp(-dt/2 H0)
        Parameters
        ----------
        hamiltonian :
            hamiltonian class
        dt : float
            time-step
        Returns
        -------
        self.BH1 : numpy array
            Exp(-dt/2 H0)
        """
        H1 = hamiltonian.h1e_mod
        I = numpy.identity(H1[0].shape[0], dtype=H1.dtype)
        print("hamiltonian.mu = {}".format(hamiltonian.mu))
        # No spin dependence for the moment.
        self.BH1 = numpy.array(
            [
                scipy.linalg.expm(-0.5 * dt * H1[0] + 0.5 * dt * hamiltonian.mu * I),
                scipy.linalg.expm(-0.5 * dt * H1[1] + 0.5 * dt * hamiltonian.mu * I),
            ]
        )

    def two_body_potentials(self, hamiltonian, iq):
        """Calculatate A and B of Eq.(13) of PRB(75)245123 for a given plane-wave vector q
        Parameters
        ----------
        hamiltonian :
            hamiltonian class
        q : float
            a plane-wave vector
        Returns
        -------
        iA : numpy array
            Eq.(13a)
        iB : numpy array
            Eq.(13b)
        """
        rho_q = hamiltonian.density_operator(iq)
        qscaled = hamiltonian.kfac * hamiltonian.qvecs[iq]

        # Due to the HS transformation, we have to do pi / 2*vol as opposed to 2*pi / vol
        piovol = math.pi / (hamiltonian.vol)
        factor = (piovol / numpy.dot(qscaled, qscaled)) ** 0.5

        # JOONHO: include a factor of 1j
        iA = 1j * factor * (rho_q + rho_q.getH())
        iB = -factor * (rho_q - rho_q.getH())
        return (iA, iB)

    def construct_force_bias(self, hamiltonian, G):
        """Compute the force bias term as in Eq.(33) of DOI:10.1002/wcms.1364
        Parameters
        ----------
        hamiltonian :
            hamiltonian class
        G : numpy array
            Green's function
        Returns
        -------
        force bias : numpy array
            -sqrt(dt) * vbias
        """
        for (i, qi) in enumerate(hamiltonian.qvecs):
            (iA, iB) = self.two_body_potentials(hamiltonian, i)
            # Deal with spin more gracefully
            self.vbias[i] = (
                iA.dot(G[0]).diagonal().sum() + iA.dot(G[1]).diagonal().sum()
            )
            self.vbias[i + self.num_vplus] = (
                iB.dot(G[0]).diagonal().sum() + iB.dot(G[1]).diagonal().sum()
            )
        return -self.sqrt_dt * self.vbias

    def construct_VHS_outofcore(self, hamiltonian, xshifted):
        """Construct the one body potential from the HS transformation
        Parameters
        ----------
        hamiltonian :
            hamiltonian class
        xshifted : numpy array
            shifited auxiliary field
        Returns
        -------
        VHS : numpy array
            the HS potential
        """
        VHS = numpy.zeros(
            (hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128
        )

        for (i, qi) in enumerate(hamiltonian.qvecs):
            (iA, iB) = self.two_body_potentials(hamiltonian, i)
            VHS = VHS + (xshifted[i] * iA).todense()
            VHS = VHS + (xshifted[i + self.num_vplus] * iB).todense()
        return VHS * self.sqrt_dt

    def construct_VHS_incore(self, hamiltonian, xshifted):
        """Construct the one body potential from the HS transformation
        Parameters
        ----------
        hamiltonian :
            hamiltonian class
        xshifted : numpy array
            shifited auxiliary field
        Returns
        -------
        VHS : numpy array
            the HS potential
        """
        VHS = numpy.zeros(
            (hamiltonian.nbasis, hamiltonian.nbasis), dtype=numpy.complex128
        )
        VHS = (
            hamiltonian.iA * xshifted[: self.num_vplus]
            + hamiltonian.iB * xshifted[self.num_vplus :]
        )
        VHS = VHS.reshape(hamiltonian.nbasis, hamiltonian.nbasis)
        return VHS * self.sqrt_dt

    def construct_force_bias_incore(self, hamiltonian, G):
        """Compute the force bias term as in Eq.(33) of DOI:10.1002/wcms.1364
        Parameters
        ----------
        hamiltonian :
            hamiltonian class
        G : numpy array
            Green's function
        Returns
        -------
        force bias : numpy array
            -sqrt(dt) * vbias
        """
        Gvec = G.reshape(2, hamiltonian.nbasis * hamiltonian.nbasis)
        self.vbias[: self.num_vplus] = (
            Gvec[0].T * hamiltonian.iA + Gvec[1].T * hamiltonian.iA
        )
        self.vbias[self.num_vplus :] = (
            Gvec[0].T * hamiltonian.iB + Gvec[1].T * hamiltonian.iB
        )
        return -self.sqrt_dt * self.vbias

    def propagate_greens_function(self, walker, B, Binv):
        if walker.stack.time_slice < walker.stack.ntime_slices:
            walker.G[0] = B[0].dot(walker.G[0]).dot(Binv[0])
            walker.G[1] = B[1].dot(walker.G[1]).dot(Binv[1])

    def two_body_propagator(self, walker, hamiltonian, force_bias=True):
        """It appliese the two-body propagator
        Parameters
        ----------
        walker :
            walker class
        hamiltonian :
            hamiltonian class
        fb : boolean
            wheter to use force bias
        Returns
        -------
        cxf : float
            the constant factor arises from mean-field shift (hard-coded for UEG for now)
        cfb : float
            the constant factor arises from the force-bias
        xshifted : numpy array
            shifited auxiliary field
        """

        # Normally distrubted auxiliary fields.
        xi = numpy.random.normal(0.0, 1.0, hamiltonian.nfields)

        # Optimal force bias.
        xbar = numpy.zeros(hamiltonian.nfields)
        if force_bias:
            rdm = one_rdm_from_G(walker.G)
            xbar = self.construct_force_bias_incore(hamiltonian, rdm)

        for i in range(hamiltonian.nfields):
            if numpy.absolute(xbar[i]) > self.fb_bound:
                if not self.nfb_trig and self.verbose:
                    print("# Rescaling force bias is triggered.")
                    print("# Warning will only be printed once per thread.")
                    print("# Bound = {}".format(self.fb_bound))
                    xb = (xbar[i].real, xbar[i].imag)
                    vb = abs(xbar[i]) / self.sqrt_dt
                    vb = (vb.real, vb.imag)
                    print("XBAR: (%f,%f)" % xb)
                    print("<v>: (%f,%f)" % vb)
                    self.nfb_trig = True
                walker.rescaled_fb = True
                xbar[i] /= numpy.absolute(xbar[i])

        xshifted = xi - xbar

        # Constant factors: Note they are not exponentiated.
        # Constant factor arising from force bias and mean field shift
        cmf = -self.sqrt_dt * xshifted.dot(self.mf_shift)
        # Constant factor arising from shifting the propability distribution.
        cfb = xi.dot(xbar) - 0.5 * xbar.dot(xbar)

        # print(xbar.dot(xbar))

        # Operator terms contributing to propagator.
        VHS = self.construct_VHS_incore(hamiltonian, xshifted)

        return (cmf, cfb, xshifted, VHS)

    def exponentiate(self, VHS, debug=False):
        """Apply exponential propagator of the HS transformation
        Parameters
        ----------
        system :
            system class
        phi : numpy array
            a state
        VHS : numpy array
            HS transformation potential
        Returns
        -------
        phi : numpy array
            Exp(VHS) * phi
        """
        # JOONHO: exact exponential
        # copy = numpy.copy(phi)
        # phi = scipy.linalg.expm(VHS).dot(copy)
        phi = numpy.identity(VHS.shape[0], dtype=numpy.complex128)
        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)

        Temp = numpy.identity(VHS.shape[0], dtype=numpy.complex128)

        for n in range(1, self.exp_nmax + 1):
            Temp = VHS.dot(Temp) / n
            phi += Temp
        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))
        return phi

    def estimate_eshift(self, walker):
        return 0.0

    def propagate_walker_free_full_rank(
        self, system, hamiltonian, walker, trial, eshift=0, force_bias=False
    ):
        """Free projection propagator
        Parameters
        ----------
        walker :
            walker class
        system :
            system class
        trial :
            trial wavefunction class
        Returns
        -------
        """

        (cmf, cfb, xmxbar, VHS) = self.two_body_propagator(
            walker, hamiltonian, force_bias=force_bias
        )
        BV = self.exponentiate(VHS)  # could use a power-series method to build this

        B = numpy.array(
            [
                numpy.einsum("ij,jj->ij", BV, self.BH1[0]),
                numpy.einsum("ij,jj->ij", BV, self.BH1[1]),
            ]
        )
        B = numpy.array(
            [
                numpy.einsum("ii,ij->ij", self.BH1[0], B[0]),
                numpy.einsum("ii,ij->ij", self.BH1[1], B[1]),
            ]
        )

        # Compute determinant ratio det(1+A')/det(1+A).
        if self.optimised:
            icur = walker.stack.time_slice // walker.stack.stack_size
            inext = (walker.stack.time_slice + 1) // walker.stack.stack_size
            if walker.stack.counter == 0:
                walker.compute_left_right(icur)
            # 1. Current walker's green's function.
            # Green's function that takes Left Right and Center
            G = walker.greens_function_left_right_no_truncation(icur, inplace=False)
            # 2. Compute updated green's function.
            walker.stack.update_new(B)
            walker.greens_function_left_right_no_truncation(icur, inplace=True)
        else:
            # Compute determinant ratio det(1+A')/det(1+A).
            # 1. Current walker's green's function.
            G = walker.greens_function(
                None, slice_ix=walker.stack.ntime_slices, inplace=False
            )
            # 2. Compute updated green's function.
            walker.stack.update_new(B)
            walker.greens_function(
                None, slice_ix=walker.stack.ntime_slices, inplace=True
            )

        # 3. Compute det(G/G')
        M0 = numpy.array(
            [
                scipy.linalg.det(G[0], check_finite=False),
                scipy.linalg.det(G[1], check_finite=False),
            ]
        )
        Mnew = numpy.array(
            [
                scipy.linalg.det(walker.G[0], check_finite=False),
                scipy.linalg.det(walker.G[1], check_finite=False),
            ]
        )

        try:
            # Could save M0 rather than recompute.
            oratio = (M0[0] * M0[1]) / (Mnew[0] * Mnew[1])

            walker.ot = 1.0
            # Constant terms are included in the walker's weight.
            (magn, phase) = cmath.polar(cmath.exp(cmf + cfb) * oratio)
            walker.weight *= magn
            walker.phase *= cmath.exp(1j * phase)
        except ZeroDivisionError:
            walker.weight = 0.0

    def propagate_walker_free_low_rank(
        self, system, walker, trial, eshift=0, force_bias=False
    ):
        """Free projection propagator
        Parameters
        ----------
        walker :
            walker class
        system :
            system class
        trial :
            trial wavefunction class
        Returns
        -------
        """

        (cmf, cfb, xmxbar, VHS) = self.two_body_propagator(
            walker, hamiltonian, force_bias=force_bias
        )
        BV = self.exponentiate(VHS)  # could use a power-series method to build this

        B = numpy.array(
            [
                numpy.einsum("ij,jj->ij", BV, self.BH1[0]),
                numpy.einsum("ij,jj->ij", BV, self.BH1[1]),
            ]
        )
        B = numpy.array(
            [
                numpy.einsum("ii,ij->ij", self.BH1[0], B[0]),
                numpy.einsum("ii,ij->ij", self.BH1[1], B[1]),
            ]
        )

        # Compute determinant ratio det(1+A')/det(1+A).
        if self.optimised:
            icur = walker.stack.time_slice // walker.stack.stack_size
            inext = (walker.stack.time_slice + 1) // walker.stack.stack_size
            if walker.stack.counter == 0:
                walker.compute_left_right(icur)
            # 1. Current walker's green's function.
            # Green's function that takes Left Right and Center
            G = walker.greens_function_left_right_no_truncation(icur, inplace=False)
            # 2. Compute updated green's function.
            walker.stack.update_new(B)
            walker.greens_function_left_right_no_truncation(icur, inplace=True)
        else:
            # Compute determinant ratio det(1+A')/det(1+A).
            # 1. Current walker's green's function.
            tix = walker.stack.ntime_slices
            G = walker.greens_function(None, slice_ix=tix, inplace=False)
            # 2. Compute updated green's function.
            walker.stack.update_new(B)
            walker.greens_function(None, slice_ix=tix, inplace=True)

        ovlp = numpy.asarray(walker.stack.ovlp).copy()
        walker.stack.update_new(B)
        ovlp_new = numpy.asarray(walker.stack.ovlp).copy()
        walker.G = walker.stack.G.copy()

        try:
            # Could save M0 rather than recompute.
            oratio = (ovlp_new[0] * ovlp_new[1]) / (ovlp[0] * ovlp[1])

            walker.ot = 1.0
            # Constant terms are included in the walker's weight.
            (magn, phase) = cmath.polar(cmath.exp(cmf + cfb) * oratio)
            walker.weight *= magn
            walker.phase *= cmath.exp(1j * phase)
        except ZeroDivisionError:
            walker.weight = 0.0

    def propagate_walker_phaseless_full_rank(
        self, hamiltonian, walker, trial, eshift=0
    ):
        # """Phaseless propagator
        # Parameters
        # ----------
        # walker :
        #     walker class
        # system :
        #     system class
        # trial :
        #     trial wavefunction class
        # Returns
        # -------
        # """

        (cmf, cfb, xmxbar, VHS) = self.two_body_propagator(walker, hamiltonian, True)
        BV = self.exponentiate(VHS)  # could use a power-series method to build this

        B = numpy.array(
            [
                numpy.einsum("ij,jj->ij", BV, self.BH1[0]),
                numpy.einsum("ij,jj->ij", BV, self.BH1[1]),
            ]
        )
        B = numpy.array(
            [
                numpy.einsum("ii,ij->ij", self.BH1[0], B[0]),
                numpy.einsum("ii,ij->ij", self.BH1[1], B[1]),
            ]
        )
        if self.optimised:
            icur = walker.stack.time_slice // walker.stack.stack_size
            inext = (walker.stack.time_slice + 1) // walker.stack.stack_size
            if walker.stack.counter == 0:
                walker.compute_left_right(icur)
            # 1. Current walker's green's function.
            # Green's function that takes Left Right and Center
            # print("walker.stack.G (before) = ", walker.G)
            G = walker.greens_function_left_right_no_truncation(icur, inplace=False)
            # print("G (before) = ", G)
            # 2. Compute updated green's function.
            walker.stack.update_new(B)
            walker.greens_function_left_right_no_truncation(icur, inplace=True)
            # print("G (after) = ", walker.G)
        else:
            # Compute determinant ratio det(1+A')/det(1+A).
            # 1. Current walker's green's function.
            tix = walker.stack.ntime_slices
            G = walker.greens_function(None, slice_ix=tix, inplace=False)
            # 2. Compute updated green's function.
            walker.stack.update_new(B)
            walker.greens_function(None, slice_ix=tix, inplace=True)

        # 3. Compute det(G/G')
        M0 = walker.M0
        Mnew = numpy.array(
            [
                scipy.linalg.det(walker.G[0], check_finite=False),
                scipy.linalg.det(walker.G[1], check_finite=False),
            ]
        )

        # Could save M0 rather than recompute.
        try:
            oratio = (M0[0] * M0[1]) / (Mnew[0] * Mnew[1])
            # Might want to cap this at some point
            hybrid_energy = cmath.log(oratio) + cfb + cmf
            Q = cmath.exp(hybrid_energy)
            expQ = self.mf_const_fac * Q
            (magn, phase) = cmath.polar(expQ)
            if not math.isinf(magn):
                # Determine cosine phase from Arg(det(1+A'(x))/det(1+A(x))).
                # Note this doesn't include exponential factor from shifting
                # propability distribution.
                dtheta = cmath.phase(cmath.exp(hybrid_energy - cfb))
                cosine_fac = max(0, math.cos(dtheta))
                walker.weight *= magn * cosine_fac
                walker.M0 = Mnew
            else:
                walker.weight = 0.0
        except ZeroDivisionError:
            walker.weight = 0.0

    def propagate_walker_phaseless_low_rank(self, hamiltonian, walker, trial, eshift=0):
        # """Phaseless propagator
        # Parameters
        # ----------
        # walker :
        #     walker class
        # system :
        #     system class
        # trial :
        #     trial wavefunction class
        # Returns
        # -------
        # """
        (cmf, cfb, xmxbar, VHS) = self.two_body_propagator(walker, hamiltonian, True)
        BV = self.exponentiate(VHS)  # could use a power-series method to build this

        B = numpy.array(
            [
                numpy.einsum("ij,jj->ij", BV, self.BH1[0]),
                numpy.einsum("ij,jj->ij", BV, self.BH1[1]),
            ]
        )
        B = numpy.array(
            [
                numpy.einsum("ii,ij->ij", self.BH1[0], B[0]),
                numpy.einsum("ii,ij->ij", self.BH1[1], B[1]),
            ]
        )

        icur = walker.stack.time_slice // walker.stack.stack_size
        #
        # local index within a stack = walker.stack.counter
        # global stack index = icur
        ovlp = numpy.asarray(walker.stack.ovlp).copy()
        walker.stack.update_new(B)
        ovlp_new = numpy.asarray(walker.stack.ovlp).copy()
        walker.G = walker.stack.G.copy()

        # Could save M0 rather than recompute.
        try:
            oratio = (ovlp_new[0] * ovlp_new[1]) / (ovlp[0] * ovlp[1])
            # Might want to cap this at some point
            hybrid_energy = cmath.log(oratio) + cfb + cmf
            Q = cmath.exp(hybrid_energy)
            expQ = self.mf_const_fac * Q
            (magn, phase) = cmath.polar(expQ)
            if not math.isinf(magn):
                # Determine cosine phase from Arg(det(1+A'(x))/det(1+A(x))).
                # Note this doesn't include exponential factor from shifting
                # propability distribution.
                dtheta = cmath.phase(cmath.exp(hybrid_energy - cfb))
                cosine_fac = max(0, math.cos(dtheta))
                walker.weight *= magn * cosine_fac
                # walker.M0 = Mnew
                walker.M0 = ovlp_new
            else:
                walker.weight = 0.0
        except ZeroDivisionError:
            walker.weight = 0.0

    def propagate_greens_function(self, walker):
        if walker.stack.time_slice < walker.stack.ntime_slices:
            walker.G[0] = self.BT[0].dot(walker.G[0]).dot(self.BTinv[0])
            walker.G[1] = self.BT[1].dot(walker.G[1]).dot(self.BTinv[1])


def unit_test():
    import cProfile

    from ipie.estimators.pw_fft import local_energy_pw_fft
    from ipie.legacy.estimators.ueg import local_energy_ueg
    from ipie.legacy.systems.ueg import UEG
    from ipie.legacy.trial_density_matrices.onebody import OneBody
    from ipie.legacy.walkers.thermal import ThermalWalker
    from ipie.qmc.comm import FakeComm
    from ipie.legacy.qmc.options import QMCOpts
    from ipie.systems.pw_fft import PW_FFT

    beta = 16.0
    dt = 0.005

    # beta = 0.5
    # dt = 0.05

    lowrank = True
    # lowrank = False

    stack_size = 10

    ecuts = [4.0, 8.0, 10.0, 12.0, 16.0, 21.0, 21.5, 32.0]
    for ecut in ecuts:
        inputs = {
            "nup": 33,
            "ndown": 33,
            "thermal": True,
            "beta": beta,
            "rs": 1.0,
            "ecut": ecut,
            "dt": dt,
            "nwalkers": 10,
            "lowrank": lowrank,
            "stack_size": stack_size,
        }

        system = UEG(inputs, True)

        qmc = QMCOpts(inputs, system, True)

        comm = FakeComm()

        trial = OneBody(comm, system, beta, dt, options=inputs, verbose=True)

        propagator = PlaneWave(system, trial, qmc, inputs, True)

        walker = ThermalWalker(
            {"stack_size": trial.stack_size, "low_rank": lowrank},
            system,
            trial,
            verbose=True,
        )
        eshift = 0.0 + 0.0j

        numpy.random.seed(7)

        pr = cProfile.Profile()
        pr.enable()
        for ts in range(0, walker.num_slices):
            propagator.propagate_walker_phaseless(
                walker=walker, system=system, trial=trial, eshift=eshift
            )

        if lowrank:
            system = PW_FFT(inputs, False)
            sort_basis = numpy.argsort(numpy.diag(system.H1[0]), kind="mergesort")
            inv_sort_basis = numpy.zeros_like(sort_basis)

            for i, idx in enumerate(sort_basis):
                inv_sort_basis[idx] = i

            mT = walker.stack.mT
            Ctrial = numpy.zeros(
                (system.nbasis, walker.stack.mT * 2), dtype=numpy.complex128
            )
            Ctrial[:, :mT] = walker.stack.CT[0][:, :mT]
            Ctrial[:, mT:] = walker.stack.CT[1][:, :mT]

            P = one_rdm_from_G(walker.G)
            # Ptmp = Ctrial[:,:mT].conj().dot(walker.stack.theta[0,:mT,:])

            # Reorder to FFT
            P[:, :, :] = P[:, inv_sort_basis, :]
            P[:, :, :] = P[:, :, inv_sort_basis]
            Theta = walker.stack.theta[:, :mT, :]
            Theta[:, :, :] = Theta[:, :, inv_sort_basis]
            Ctrial = Ctrial[inv_sort_basis, :]

            print(
                "E = {}".format(
                    local_energy_pw_fft(system, G=P, Ghalf=Theta, trial=Ctrial)
                )
            )
        else:
            P = one_rdm_from_G(walker.G)
            print(numpy.diag(walker.G[0].real))
            print("weight = {}".format(walker.weight))
            print("E = {}".format(local_energy_ueg(system, P)))

        pr.disable()
        pr.print_stats(sort="tottime")


if __name__ == "__main__":
    unit_test()
