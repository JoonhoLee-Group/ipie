import cmath
import copy
import math

import numpy
import scipy.linalg

from ipie.legacy.estimators.thermal import one_rdm_from_G


class ThermalDiscrete(object):
    def __init__(self, ham, trial, qmc, options={}, verbose=False, lowrank=False):

        if verbose:
            print("# Parsing discrete propagator input options.")
            print("# Using continuous Hubbar--Stratonovich transformations.")
        self.free_projection = options.get("free_projection", False)
        self.nstblz = qmc.nstblz
        self.hs_type = "discrete"
        self.charge_decomp = options.get("charge_decomposition", False)
        if verbose:
            if self.charge_decomp:
                print("# Using charge decomposition.")
            else:
                print("# Using spin decomposition.")
        # [field,spin]
        if self.charge_decomp:
            self.gamma = numpy.arccosh(numpy.exp(-0.5 * qmc.dt * ham.U + 0j))
            self.auxf = numpy.array(
                [
                    [numpy.exp(self.gamma), numpy.exp(self.gamma)],
                    [numpy.exp(-self.gamma), numpy.exp(-self.gamma)],
                ]
            )
            # e^{-gamma x}
            self.aux_wfac = numpy.exp(0.5 * qmc.dt * ham.U) * numpy.array(
                [numpy.exp(-self.gamma), numpy.exp(self.gamma)]
            )
        else:
            self.gamma = numpy.arccosh(numpy.exp(0.5 * qmc.dt * ham.U))
            self.auxf = numpy.array(
                [
                    [numpy.exp(self.gamma), numpy.exp(-self.gamma)],
                    [numpy.exp(-self.gamma), numpy.exp(self.gamma)],
                ]
            )
            self.aux_wfac = numpy.array([1.0, 1.0])
        if not ham.symmetric:
            self.auxf = self.auxf * numpy.exp(-0.5 * qmc.dt * ham.U)
        # Account for potential shift in chemical potential
        sign = 1 if ham._alt_convention else -1
        self.dmu = sign * (ham.mu - trial.mu)
        self.auxf *= numpy.exp(-qmc.dt * (self.dmu))
        if abs(self.dmu) > 1e-16:
            self._mu = trial.mu
            if verbose:
                print(
                    "# Chemical potential shift (mu_T-mu): {}".format(-sign * self.dmu)
                )
        else:
            self._mu = ham.mu
        self.delta = self.auxf - 1
        dt = qmc.dt
        dmat_up = scipy.linalg.expm(-dt * (ham.H1[0]))
        dmat_down = scipy.linalg.expm(-dt * (ham.H1[1]))
        dmat = numpy.array([dmat_up, dmat_down])
        self.construct_one_body_propagator(ham, self._mu, dt)
        self.BT_BP = None
        self.BT = trial.dmat
        self.BT_inv = trial.dmat_inv
        if self.charge_decomp:
            dtype = numpy.complex128
        else:
            dtype = trial.dmat.dtype
        self.BV = numpy.zeros((2, trial.dmat.shape[-1]), dtype=dtype)
        if self.free_projection:
            self.propagate_walker = self.propagate_walker_free
        else:
            self.propagate_walker = self.propagate_walker_constrained

    def construct_one_body_propagator(self, ham, mu, dt):
        """Construct the one-body propagator Exp(-dt/2 H0)
        Parameters
        ----------
        ham :
            ham class
        dt : float
            time-step
        Returns
        -------
        self.BH1 : numpy array
            Exp(-dt H0)
        """
        H1 = ham.H1
        I = numpy.identity(H1[0].shape[0], dtype=H1.dtype)
        # No spin dependence for the moment.
        sign = 1 if ham._alt_convention else -1
        self.BH1 = numpy.array(
            [
                scipy.linalg.expm(-dt * (H1[0] + sign * mu * I)),
                scipy.linalg.expm(-dt * (H1[1] + sign * mu * I)),
            ]
        )

    def update_greens_function_simple(self, walker, time_slice):
        walker.construct_greens_function_stable(time_slice)

    def update_greens_function(self, walker, i, xi):
        for spin in [0, 1]:
            g = walker.G[spin, :, i]
            gbar = -walker.G[spin, i, :]
            gbar[i] += 1
            denom = 1 + (1 - g[i]) * self.delta[xi, spin]
            walker.G[spin] = (
                walker.G[spin]
                - self.delta[xi, spin] * numpy.einsum("i,j->ij", g, gbar) / denom
            )

    def propagate_greens_function(self, walker):
        if walker.stack.time_slice < walker.stack.ntime_slices:
            walker.G[0] = self.BT[0].dot(walker.G[0]).dot(self.BT_inv[0])
            walker.G[1] = self.BT[1].dot(walker.G[1]).dot(self.BT_inv[1])

    def calculate_overlap_ratio(self, walker, i):
        R1_up = 1 + (1 - walker.G[0, i, i]) * self.delta[0, 0]
        R1_dn = 1 + (1 - walker.G[1, i, i]) * self.delta[0, 1]
        R2_up = 1 + (1 - walker.G[0, i, i]) * self.delta[1, 0]
        R2_dn = 1 + (1 - walker.G[1, i, i]) * self.delta[1, 1]
        return 0.5 * numpy.array([R1_up * R1_dn, R2_up * R2_dn])

    def estimate_eshift(self, walker):
        oratio = self.calculate_overlap_ratio(walker, 0)
        return sum(oratio)

    def propagate_walker_constrained(self, system, walker, time_slice, eshift=0):
        for i in range(0, system.nbasis):
            probs = self.calculate_overlap_ratio(walker, i)
            phaseless_ratio = numpy.maximum(probs.real, [0, 0])
            norm = sum(phaseless_ratio)
            r = numpy.random.random()
            if norm > 0:
                walker.weight = walker.weight * norm * numpy.exp(eshift)
                # if walker.weight > walker.total_weight * 0.10:
                # walker.weight = walker.total_weight * 0.10
                if r < phaseless_ratio[0] / norm:
                    xi = 0
                else:
                    xi = 1
                self.update_greens_function(walker, i, xi)
                self.BV[0, i] = self.auxf[xi, 0]
                self.BV[1, i] = self.auxf[xi, 1]
            else:
                walker.weight = 0
        B = numpy.einsum("ki,kij->kij", self.BV, self.BH1)
        walker.stack.update(B)
        # Need to recompute Green's function from scratch before we propagate it
        # to the next time slice due to stack structure.
        if walker.stack.time_slice % self.nstblz == 0:
            walker.greens_function(None, walker.stack.time_slice - 1)
        self.propagate_greens_function(walker)

    def propagate_walker_free(self, system, walker, time_slice, eshift):
        fields = numpy.random.randint(0, 2, system.nbasis)
        self.BV[0] = numpy.array([self.auxf[xi, 0] for xi in fields])
        self.BV[1] = numpy.array([self.auxf[xi, 1] for xi in fields])
        # Vsii Tsij
        B = numpy.einsum("ki,kij->kij", self.BV, self.BH1)
        wfac = 1.0 + 0j
        for xi in fields:
            wfac *= self.aux_wfac[xi]
            # Compute determinant ratio det(1+A')/det(1+A).
            # 1. Current walker's green's function.
        G = walker.greens_function(
            None, slice_ix=walker.stack.ntime_slices, inplace=False
        )
        # 2. Compute updated green's function.
        walker.stack.update_new(B)
        walker.greens_function(None, slice_ix=walker.stack.ntime_slices, inplace=True)

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
            oratio = wfac * (M0[0] * M0[1]) / (Mnew[0] * Mnew[1])

            walker.ot = 1.0
            # Constant terms are included in the walker's weight.
            (magn, phase) = cmath.polar(oratio)
            walker.weight *= magn
            walker.phase *= cmath.exp(1j * phase)
        except ZeroDivisionError:
            walker.weight = 0.0
        # Need to recompute Green's function from scratch before we propagate it
        # to the next time slice due to stack structure.


class HubbardContinuous(object):
    """Propagator for continuous HS transformation, specialised for Hubbard model.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pie.qmc.options.QMCOpts`
        QMC options.
    system : :class:`pie.system.System`
        System object.
    trial : :class:`pie.trial_wavefunctioin.Trial`
        Trial wavefunction object.
    verbose : bool
        If true print out more information during setup.
    """

    def __init__(self, system, trial, qmc, options={}, verbose=False):
        if verbose:
            print("# Parsing continuous propagator input options.")
        self.hs_type = "hubbard_continuous"
        self.free_projection = options.get("free_projection", False)
        self.ffts = options.get("ffts", False)
        self.nstblz = qmc.nstblz
        self.btk = numpy.exp(-0.5 * qmc.dt * system.eks)
        model = system.__class__.__name__
        self.dt = qmc.dt
        self.iu_fac = 1j * system.U**0.5
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j * self.sqrt_dt
        # optimal mean-field shift for the hubbard model
        P = one_rdm_from_G(trial.G)
        # Mean field shifts (2,nchol_vec).
        self.mf_shift = self.construct_mean_field_shift(system, P)
        if verbose:
            print(
                "# Absolute value of maximum component of mean field shift: "
                "{:13.8e}.".format(numpy.max(numpy.abs(self.mf_shift)))
            )
        self.mf_core = 0.5 * numpy.dot(self.mf_shift, self.mf_shift)
        # if self.ffts:
        # self.kinetic = kinetic_kspace
        # else:
        # self.kinetic = kinetic_real
        if verbose:
            print("# Finished propagator input options.")

    def construct_one_body_propagator(self, system, dt):
        # \sum_gamma v_MF^{gamma} v^{\gamma}
        vi1b = self.iu_fac * numpy.diag(self.mf_shift)
        I = numpy.identity(system.H1[0].shape[0], dtype=system.H1.dtype)
        muN = system.mu * I
        sign = 1 if system._alt_convention else -1
        H1 = system.h1e_mod - numpy.array([vi1b - sign * muN, vi1b - sign * muN])
        self.BH1 = numpy.array(
            [scipy.linalg.expm(-0.5 * dt * H1[0]), scipy.linalg.expm(-0.5 * dt * H1[1])]
        )

    def construct_mean_field_shift(self, system, P):
        #  i sqrt{U} < n_{iup} + n_{idn} >_MF
        return self.iu_fac * (numpy.diag(P[0]) + numpy.diag(P[1]))

    def construct_force_bias(self, system, P, trial):
        #  i sqrt{U} < n_{iup} + n_{idn} > - mf_shift
        vbias = self.iu_fac * (numpy.diag(P[0]) + numpy.diag(P[1]))
        return -self.sqrt_dt * (vbias - self.mf_shift)

    def construct_VHS(self, system, shifted):
        # Note factor of i included in v_i
        # B_V(x-\bar{x}) = e^{\sqrt{dt}*(x-\bar{x})\hat{v}_i}
        # v_i = n_{iu} + n_{id}
        return numpy.diag(self.sqrt_dt * self.iu_fac * shifted)
