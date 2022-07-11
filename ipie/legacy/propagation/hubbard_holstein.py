import cmath
import copy
import math

import numpy
import scipy.linalg

from ipie.legacy.propagation.operations import local_energy_bound
from ipie.legacy.trial_wavefunction.harmonic_oscillator import (
    HarmonicOscillator, HarmonicOscillatorMomentum)
from ipie.legacy.utils.fft import fft_wavefunction, ifft_wavefunction
from ipie.legacy.walkers.multi_ghf import MultiGHFWalker
from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.propagation.operations import kinetic_real
from ipie.utils.linalg import reortho


def arccosh(y):  # it works even when y is complex
    gamma = cmath.log(y - cmath.sqrt(y * y - 1))
    return gamma


class HirschDMC(object):
    """Propagator for discrete HS transformation plus phonon propagation.

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
            print("# Parsing discrete propagator input options.")
            print("# Using discrete Hubbard--Stratonovich transformation.")

        if trial.type == "GHF":
            self.bt2 = scipy.linalg.expm(-0.5 * qmc.dt * system.T[0])
        else:
            self.bt2 = numpy.array(
                [
                    scipy.linalg.expm(-0.5 * qmc.dt * system.T[0]),
                    scipy.linalg.expm(-0.5 * qmc.dt * system.T[1]),
                ]
            )

        if trial.type == "GHF" and trial.bp_wfn is not None:
            self.BT_BP = scipy.linalg.block_diag(self.bt2, self.bt2)
            self.back_propagate = back_propagate_ghf
        else:
            self.BT_BP = self.bt2
            self.back_propagate = back_propagate

        self.nstblz = qmc.nstblz
        self.btk = numpy.exp(-0.5 * qmc.dt * system.eks)
        self.ffts = options.get("ffts", False)
        self.hs_type = "discrete"
        self.free_projection = options.get("free_projection", False)

        self.symmetric_trotter = options.get("symmetric_trotter", False)
        if verbose:
            print("# symmetric_trotter is {}".format(self.symmetric_trotter))

        Ueff = system.U

        self.lang_firsov = inputs.get("lang_firsov", False)
        self.gamma_lf = 0.0
        if self.lang_firsov:
            self.gamma_lf = system.gamma_lf
            Ueff = system.Ueff

        if verbose:
            print("# Ueff = {}".format(Ueff))

        self.sorella = options.get("sorella", False)
        self.charge = options.get("charge", False)

        if self.sorella == True:
            self.charge = True

        if not self.charge:
            self.gamma = arccosh(numpy.exp(0.5 * qmc.dt * Ueff))
            if verbose:
                print("# Spin decomposition is used")
            # field by spin
            self.auxf = numpy.array(
                [
                    [numpy.exp(self.gamma), numpy.exp(-self.gamma)],
                    [numpy.exp(-self.gamma), numpy.exp(self.gamma)],
                ]
            )
            self.auxf = self.auxf * numpy.exp(-0.5 * qmc.dt * Ueff)

        else:
            if verbose:
                print("# Charge decomposition is used")

            self.gamma = arccosh(numpy.exp(-0.5 * qmc.dt * Ueff))

            if self.sorella:
                if verbose:
                    print("# Sorella decomposition is used")
                self.charge_factor = numpy.array(
                    [numpy.exp(-0.5 * self.gamma), numpy.exp(0.5 * self.gamma)]
                )
            else:
                self.charge_factor = numpy.array(
                    [numpy.exp(-self.gamma), numpy.exp(self.gamma)]
                ) * numpy.exp(0.5 * qmc.dt * Ueff)

            if verbose:
                print("# charge_factor = {}".format(self.charge_factor))

            # field by spin
            self.auxf = numpy.array(
                [
                    [numpy.exp(self.gamma), numpy.exp(self.gamma)],
                    [numpy.exp(-self.gamma), numpy.exp(-self.gamma)],
                ]
            )
            if not self.sorella:
                self.auxf = self.auxf * numpy.exp(-0.5 * qmc.dt * Ueff)

        self.hybrid = False

        self.dt = qmc.dt
        self.sqrtdt = math.sqrt(qmc.dt)
        self.delta = self.auxf - 1

        if self.free_projection:
            self.propagate_walker = self.propagate_walker_free
        else:
            self.propagate_walker = self.propagate_walker_constrained

        if trial.symmetrize:
            self.calculate_overlap_ratio = calculate_overlap_ratio_multi_det
            self.update_greens_function = self.update_greens_function_mult
        else:
            self.calculate_overlap_ratio = calculate_overlap_ratio_single_det
            self.update_greens_function = self.update_greens_function_uhf

        if self.ffts:
            self.kinetic = kinetic_kspace
        else:
            self.kinetic = kinetic_real

        shift = trial.shift.copy()
        if verbose:
            if len(trial.psi.shape) == 3:
                print("# Shift in propagation = {}".format(shift[0, :3]))
            else:
                print("# Shift in propagation = {}".format(shift[:3]))

        if len(trial.psi.shape) == 3:
            self.boson_trial = HarmonicOscillator(
                m=system.m, w=system.w0, order=0, shift=shift[0, :]
            )
            self.eshift_boson = self.boson_trial.local_energy(shift[0, :])
        else:
            self.boson_trial = HarmonicOscillator(
                m=system.m, w=system.w0, order=0, shift=shift
            )
            self.eshift_boson = self.boson_trial.local_energy(shift)

        self.eshift_boson = self.eshift_boson.real

        if verbose:
            print("# Finished setting up propagator.")

    def update_greens_function_uhf(self, walker, trial, i, nup):
        """Fast update of walker's Green's function for RHF/UHF walker.

        Parameters
        ----------
        walker : :class:`pie.walkers.SingleDet`
            Walker's wavefunction.
        trial : :class:`pie.trial_wavefunction`
            Trial wavefunction.
        i : int
            Basis index.
        nup : int
            Number of up electrons.
        """

        ndown = walker.phi.shape[1] - nup

        vup = trial.psi.conj()[i, :nup]
        uup = walker.phi[i, :nup]
        q = numpy.dot(walker.inv_ovlp[0], vup)
        walker.G[0][i, i] = numpy.dot(uup, q)
        vdown = trial.psi.conj()[i, nup:]
        udown = walker.phi[i, nup:]
        if ndown > 0:
            q = numpy.dot(walker.inv_ovlp[1], vdown)
            walker.G[1][i, i] = numpy.dot(udown, q)

    def update_greens_function_mult(self, walker, trial, i, nup):
        """Fast update of walker's Green's function for multi RHF/UHF walker.

        Parameters
        ----------
        walker : :class:`pie.walkers.SingleDet`
            Walker's wavefunction.
        trial : :class:`pie.trial_wavefunction`
            Trial wavefunction.
        i : int
            Basis index.
        nup : int
            Number of up electrons.
        """

        ndown = walker.phi.shape[1] - nup

        if len(trial.psi.shape) == 3:

            for ix in range(trial.nperms):
                psi = trial.psi[ix, :, :].copy()
                vup = psi.conj()[i, :nup]

                uup = walker.phi[i, :nup]

                q = numpy.dot(walker.inv_ovlp[0][ix], vup)

                walker.Gi[ix, 0, i, i] = numpy.dot(uup, q)

                vdown = psi.conj()[i, nup:]
                udown = walker.phi[i, nup:]

                if ndown > 0:
                    q = numpy.dot(walker.inv_ovlp[1][ix], vdown)
                    walker.Gi[ix, 1, i, i] = numpy.dot(udown, q)

        else:
            for ix, perm in enumerate(trial.perms):
                psi = trial.psi[perm, :].copy()
                vup = psi.conj()[i, :nup]

                uup = walker.phi[i, :nup]

                q = numpy.dot(walker.inv_ovlp[0][ix], vup)

                walker.Gi[ix, 0, i, i] = numpy.dot(uup, q)

                vdown = psi.conj()[i, nup:]
                udown = walker.phi[i, nup:]

                if ndown > 0:
                    q = numpy.dot(walker.inv_ovlp[1][ix], vdown)
                    walker.Gi[ix, 1, i, i] = numpy.dot(udown, q)

    def update_greens_function_ghf(self, walker, trial, i, nup):
        """Update of walker's Green's function for UHF walker.

        Parameters
        ----------
        walker : :class:`pie.walkers.SingleDet`
            Walker's wavefunction.
        trial : :class:`pie.trial_wavefunction`
            Trial wavefunction.
        i : int
            Basis index.
        nup : int
            Number of up electrons.
        """
        walker.greens_function(trial)

    def two_body(self, walker, system, trial):
        r"""Propagate by potential term using discrete HS transform.

        Parameters
        ----------
        walker : :class:`pie.walker` object
            Walker object to be updated. On output we have acted on phi by
            B_V(x) and updated the weight appropriately. Updates inplace.
        system : :class:`pie.system.System`
            System object.
        trial : :class:`pie.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """
        # Construct random auxilliary field.
        delta = self.delta
        nup = system.nup
        soffset = walker.phi.shape[0] - system.nbasis
        for i in range(0, system.nbasis):
            self.update_greens_function(walker, trial, i, nup)
            # Ratio of determinants for the two choices of auxilliary fields
            probs = self.calculate_overlap_ratio(walker, delta, trial, i)
            if self.charge:
                probs *= self.charge_factor

            if self.sorella:
                const = (
                    -self.gamma
                    * system.g
                    * math.sqrt(2.0 * system.m * system.w0)
                    / system.U
                    * walker.X[i]
                )
                factor = numpy.array([numpy.exp(const), numpy.exp(-const)])
                probs *= factor

            # issues here with complex numbers?
            phaseless_ratio = numpy.maximum(probs.real, [0, 0])
            norm = sum(phaseless_ratio)
            r = numpy.random.random()
            # Is this necessary?
            # todo : mirror correction
            if norm > 0:
                walker.weight = walker.weight * norm
                if r < phaseless_ratio[0] / norm:
                    xi = 0
                else:
                    xi = 1
                vtup = walker.phi[i, :nup] * delta[xi, 0]
                vtdown = walker.phi[i + soffset, nup:] * delta[xi, 1]
                walker.phi[i, :nup] = walker.phi[i, :nup] + vtup
                walker.phi[i + soffset, nup:] = walker.phi[i + soffset, nup:] + vtdown
                walker.update_overlap(probs, xi, trial.coeffs)
                if walker.field_configs is not None:
                    walker.field_configs.push(xi)
                walker.update_inverse_overlap(trial, vtup, vtdown, i)
            else:
                walker.weight = 0
                return

    def acceptance(self, posold, posnew, driftold, driftnew, trial):

        gfratio = numpy.exp(
            -numpy.sum((posold - posnew - driftnew) ** 2 / (2 * self.dt))
            + numpy.sum((posnew - posold - driftold) ** 2 / (2 * self.dt))
        )

        ratio = trial.value(posnew) ** 2 / trial.value(posold) ** 2

        return ratio * gfratio

    def boson_importance_sampling(self, walker, system, trial, dt):

        sqrtdt = numpy.sqrt(dt)

        phiold = trial.value(walker)

        # Drift+diffusion
        driftold = (dt / system.m) * trial.gradient(walker)

        if self.sorella:
            Ev = (
                0.5
                * system.m
                * system.w0**2
                * (1.0 - 2.0 * system.g**2 / (system.w0 * system.U))
                * numpy.sum(walker.X * walker.X)
            )
            Ev2 = (
                -0.5
                * numpy.sqrt(2.0 * system.m * system.w0)
                * system.g
                * numpy.sum(walker.X)
            )
            lap = trial.laplacian(walker)
            Ek = 0.5 / (system.m) * numpy.sum(lap * lap)
            elocold = Ev + Ev2 + Ek
        else:
            elocold = trial.bosonic_local_energy(walker)

        elocold = numpy.real(elocold)

        dX = numpy.random.normal(
            loc=0.0, scale=sqrtdt / numpy.sqrt(system.m), size=(system.nbasis)
        )
        Xnew = walker.X + dX + driftold

        walker.X = Xnew.copy()

        phinew = trial.value(walker)
        lap = trial.laplacian(walker)
        walker.Lap = lap

        # Change weight
        if self.sorella:
            Ev = (
                0.5
                * system.m
                * system.w0**2
                * (1.0 - 2.0 * system.g**2 / (system.w0 * system.U))
                * numpy.sum(walker.X * walker.X)
            )
            Ev2 = (
                -0.5
                * numpy.sqrt(2.0 * system.m * system.w0)
                * system.g
                * numpy.sum(walker.X)
            )
            lap = trial.laplacian(walker)
            Ek = 0.5 / (system.m) * numpy.sum(lap * lap)
            eloc = Ev + Ev2 + Ek
        else:
            eloc = trial.bosonic_local_energy(walker)

        eloc = numpy.real(eloc)
        walker.ot *= phinew / phiold

        walker.weight *= math.exp(-0.5 * dt * (eloc + elocold - 2 * self.eshift_boson))

    def kinetic_importance_sampling(self, walker, system, trial, dt):
        r"""Propagate by the kinetic term by direct matrix multiplication.

        Parameters
        ----------
        walker : :class:`pie.walker`
            Walker object to be updated. On output we have acted on phi by
            B_{T/2} and updated the weight appropriately. Updates inplace.
        system : :class:`pie.system.System`
            System object.
        trial : :class:`pie.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """
        # bt2 = [scipy.linalg.expm(-dt*system.T[0]), scipy.linalg.expm(-dt*system.T[1])]
        # kinetic_real(walker.phi, system, bt2, H1diag=False)

        if not self.sorella:
            const = (-system.g * cmath.sqrt(system.m * system.w0 * 2.0)) * (-dt)
            const = const.real
            nX = [walker.X, walker.X]
            # Veph = [numpy.diag( numpy.exp(const * nX[0]) ),numpy.diag( numpy.exp(const * nX[1]) )]
            # kinetic_real(walker.phi, system, Veph, H1diag=True)
            TV = [
                scipy.linalg.expm(-dt * system.T[0] + numpy.diag(const * nX[0])),
                scipy.linalg.expm(-dt * system.T[1] + numpy.diag(const * nX[1])),
            ]
            # print(walker.phi.dtype, walker.X.dtype, const)
            kinetic_real(walker.phi, system, TV, H1diag=False)

        # Update inverse overlap
        walker.inverse_overlap(trial)
        # Update walker weight
        ot_new = walker.calc_otrial(trial)

        ratio = ot_new / walker.ot
        phase = cmath.phase(ratio)

        if abs(phase) < 0.5 * math.pi:
            (magn, phase) = cmath.polar(ratio)
            cosine_fac = max(0, math.cos(phase))
            walker.weight *= magn * cosine_fac
            walker.ot = ot_new
        else:
            walker.ot = ot_new
            walker.weight = 0.0

    def propagate_walker_constrained(
        self, walker, system, trial, eshift, rho=None, X=None
    ):
        r"""Wrapper function for propagation using discrete transformation

        The discrete transformation allows us to split the application of the
        projector up a bit more, which allows up to make use of fast matrix
        update routines since only a row might change.

        Parameters
        ----------
        walker : :class:`pie.walker` object
            Walker object to be updated. On output we have acted on phi by
            B_V(x) and updated the weight appropriately. Updates inplace.
        system : :class:`pie.system.System`
            System object.
        trial : :class:`pie.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """
        if self.symmetric_trotter:
            if abs(walker.weight.real) > 0:
                self.boson_importance_sampling(walker, system, trial, self.dt / 2.0)
            if abs(walker.weight) > 0:
                self.kinetic_importance_sampling(walker, system, trial, self.dt / 2.0)
            if abs(walker.weight) > 0:
                self.two_body(walker, system, trial)  # hard-coded to do self.dt
            if abs(walker.weight.real) > 0:
                self.kinetic_importance_sampling(walker, system, trial, self.dt / 2.0)
            if abs(walker.weight.real) > 0:
                self.boson_importance_sampling(walker, system, trial, self.dt / 2.0)
        else:
            if abs(walker.weight) > 0:
                self.kinetic_importance_sampling(walker, system, trial, self.dt / 2.0)
            if abs(walker.weight) > 0:
                self.two_body(walker, system, trial)  # hard-coded to do self.dt
            if abs(walker.weight.real) > 0:
                self.kinetic_importance_sampling(walker, system, trial, self.dt / 2.0)
            if abs(walker.weight.real) > 0:
                self.boson_importance_sampling(walker, system, trial, self.dt)

    def boson_free_propagation(self, walker, system, trial, eshift):
        # Change weight
        pot = 0.25 * system.m * system.w0 * system.w0 * numpy.sum(walker.X * walker.X)
        pot = pot.real
        walker.weight *= math.exp(-self.dt * pot)

        psiold = self.boson_trial.value(walker.X)

        dX = numpy.random.normal(
            loc=0.0, scale=self.sqrtdt / numpy.sqrt(system.m), size=(system.nbasis)
        )
        Xnew = walker.X + dX

        walker.X = Xnew.copy()

        lap = self.boson_trial.laplacian(walker.X)
        walker.Lap = lap

        psinew = self.boson_trial.value(walker.X)

        pot = 0.25 * system.m * system.w0 * system.w0 * numpy.sum(walker.X * walker.X)
        pot = pot.real
        walker.weight *= math.exp(-self.dt * pot)

    def propagate_walker_free(self, walker, system, trial, eshift):
        r"""Propagate walker without imposing constraint.

        Uses single-site updates for potential term.

        Parameters
        ----------
        walker : :class:`pie.walker` object
            Walker object to be updated. On output we have acted on phi by
            B_V(x) and updated the weight appropriately. Updates inplace.
        system : :class:`pie.system.System`
            System object.
        trial : :class:`pie.trial_wavefunctioin.Trial`
            Trial wavefunction object.
        """
        self.boson_free_propagation(walker, system, self.boson_trial, eshift)

        kinetic_real(walker.phi, system, self.bt2)

        const = system.g * cmath.sqrt(system.m * system.w0 * 2.0) * self.dt / 2.0
        Veph = [
            numpy.diag(numpy.exp(const * walker.X)),
            numpy.diag(numpy.exp(const * walker.X)),
        ]
        kinetic_real(walker.phi, system, Veph, H1diag=True)

        delta = self.delta
        nup = system.nup
        for i in range(0, system.nbasis):
            if abs(walker.weight) > 0:
                r = numpy.random.random()
                if r < 0.5:
                    xi = 0
                else:
                    xi = 1
                vtup = walker.phi[i, :nup] * delta[xi, 0]
                vtdown = walker.phi[i, nup:] * delta[xi, 1]
                walker.phi[i, :nup] = walker.phi[i, :nup] + vtup
                walker.phi[i, nup:] = walker.phi[i, nup:] + vtdown
                if self.charge:
                    walker.weight *= self.charge_factor[xi]

        kinetic_real(walker.phi, system, Veph, H1diag=True)

        kinetic_real(walker.phi, system, self.bt2)

        walker.inverse_overlap(trial)

        # Update walker weight
        walker.ot = walker.calc_otrial(trial.psi) * self.boson_trial.value(walker.X)

        walker.greens_function(trial)
        # Constant terms are included in the walker's weight.
        # (magn, dtheta) = cmath.polar(cmath.exp(cmf+self.dt*eshift))
        # walker.weight *= magn
        # walker.phase *= cmath.exp(1j*dtheta)


def calculate_overlap_ratio_multi_ghf(walker, delta, trial, i):
    """Calculate overlap ratio for single site update with GHF trial.

    Parameters
    ----------
    walker : walker object
        Walker to be updated.
    delta : :class:`numpy.ndarray`
        Delta updates for single spin flip.
    trial : trial wavefunctio object
        Trial wavefunction.
    i : int
        Basis index.
    """
    nbasis = trial.psi.shape[1] // 2
    for (idx, G) in enumerate(walker.Gi):
        guu = G[i, i]
        gdd = G[i + nbasis, i + nbasis]
        gud = G[i, i + nbasis]
        gdu = G[i + nbasis, i]
        walker.R[idx, 0] = (1 + delta[0, 0] * guu) * (1 + delta[0, 1] * gdd) - delta[
            0, 0
        ] * gud * delta[0, 1] * gdu
        walker.R[idx, 1] = (1 + delta[1, 0] * guu) * (1 + delta[1, 1] * gdd) - delta[
            1, 0
        ] * gud * delta[1, 1] * gdu
    R = numpy.einsum("i,ij,i->j", trial.coeffs, walker.R, walker.ots) / walker.ot
    return 0.5 * numpy.array([R[0], R[1]])


def calculate_overlap_ratio_multi_det(walker, delta, trial, i):
    """Calculate overlap ratio for single site update with multi-det trial.

    Parameters
    ----------
    walker : walker object
        Walker to be updated.
    delta : :class:`numpy.ndarray`
        Delta updates for single spin flip.
    trial : trial wavefunctio object
        Trial wavefunction.
    i : int
        Basis index.
    """
    for (idx, G) in enumerate(walker.Gi):
        walker.R[idx, 0] = (1 + delta[0][0] * G[0][i, i]) * (
            1 + delta[0][1] * G[1][i, i]
        )
        walker.R[idx, 1] = (1 + delta[1][0] * G[0][i, i]) * (
            1 + delta[1][1] * G[1][i, i]
        )

    denom = numpy.sum(walker.weights)
    R = numpy.einsum("i,ix->x", walker.weights, walker.R) / denom
    # spin_prod = numpy.einsum('ikj,ji->ikj',walker.R,walker.ots)
    # R = numpy.einsum('i,ij->j',trial.coeffs,spin_prod[:,:,0]*spin_prod[:,:,1])/walker.ot
    return 0.5 * numpy.array([R[0], R[1]])


def calculate_overlap_ratio_single_det(walker, delta, trial, i):
    """Calculate overlap ratio for single site update with UHF trial.

    Parameters
    ----------
    walker : walker object
        Walker to be updated.
    delta : :class:`numpy.ndarray`
        Delta updates for single spin flip.
    trial : trial wavefunctio object
        Trial wavefunction.
    i : int
        Basis index.
    """
    R1 = (1 + delta[0][0] * walker.G[0][i, i]) * (1 + delta[0][1] * walker.G[1][i, i])
    R2 = (1 + delta[1][0] * walker.G[0][i, i]) * (1 + delta[1][1] * walker.G[1][i, i])
    return 0.5 * numpy.array([R1, R2])


def construct_propagator_matrix(system, BT2, config, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    For use with discrete transformation.

    Parameters
    ----------
    system : class
        System class.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    config : numpy array
        Auxiliary field configuration.
    conjt : bool
        If true return Hermitian conjugate of matrix.

    Returns
    -------
    B : :class:`numpy.ndarray`
        Full projector matrix.
    """
    bv_up = numpy.diag(numpy.array([system.auxf[xi, 0] for xi in config]))
    bv_down = numpy.diag(numpy.array([system.auxf[xi, 1] for xi in config]))
    Bup = BT2[0].dot(bv_up).dot(BT2[0])
    Bdown = BT2[1].dot(bv_down).dot(BT2[1])

    if conjt:
        return numpy.array([Bup.conj().T, Bdown.conj().T])
    else:
        return numpy.array([Bup, Bdown])


def construct_propagator_matrix_ghf(system, BT2, config, conjt=False):
    """Construct the full projector from a configuration of auxiliary fields.

    For use with GHF trial wavefunction.

    Parameters
    ----------
    system : class
        System class.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    config : numpy array
        Auxiliary field configuration.
    conjt : bool
        If true return Hermitian conjugate of matrix.

    Returns
    -------
    B : :class:`numpy.ndarray`
        Full projector matrix.
    """
    bv_up = numpy.diag(numpy.array([system.auxf[xi, 0] for xi in config]))
    bv_down = numpy.diag(numpy.array([system.auxf[xi, 1] for xi in config]))
    BV = scipy.linalg.block_diag(bv_up, bv_down)
    B = BT2.dot(BV).dot(BT2)

    if conjt:
        return B.conj().T
    else:
        return B


def back_propagate(system, psi, trial, nstblz, BT2, dt):
    r"""Perform back propagation for UHF style wavefunction.

    Parameters
    ---------
    system : system object in general.
        Container for model input options.
    psi : :class:`pie.walkers.Walkers` object
        CPMC wavefunction.
    trial : :class:`pie.trial_wavefunction.X' object
        Trial wavefunction class.
    nstblz : int
        Number of steps between GS orthogonalisation.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    dt : float
        Timestep.

    Returns
    -------
    psi_bp : list of :class:`pie.walker.Walker` objects
        Back propagated list of walkers.
    """

    psi_bp = [SingleDetWalker({}, system, trial, index=w) for w in range(len(psi))]
    nup = system.nup
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, c) in enumerate(w.field_configs.get_block()[0][::-1]):
            B = construct_propagator_matrix(system, BT2, c, conjt=True)
            psi_bp[iw].phi[:, :nup] = B[0].dot(psi_bp[iw].phi[:, :nup])
            psi_bp[iw].phi[:, nup:] = B[1].dot(psi_bp[iw].phi[:, nup:])
            if i != 0 and i % nstblz == 0:
                psi_bp[iw].reortho(trial)
    return psi_bp


def back_propagate_ghf(system, psi, trial, nstblz, BT2, dt):
    r"""Perform back propagation for GHF style wavefunction.

    Parameters
    ---------
    system : system object in general.
        Container for model input options.
    psi : :class:`pie.walkers.Walkers` object
        CPMC wavefunction.
    trial : :class:`pie.trial_wavefunction.X' object
        Trial wavefunction class.
    nstblz : int
        Number of steps between GS orthogonalisation.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    dt : float
        Timestep.

    Returns
    -------
    psi_bp : list of :class:`pie.walker.Walker` objects
        Back propagated list of walkers.
    """
    psi_bp = [
        MultiGHFWalker({}, system, trial, index=w, weights="ones", wfn0="GHF")
        for w in range(len(psi))
    ]
    for (iw, w) in enumerate(psi):
        # propagators should be applied in reverse order
        for (i, c) in enumerate(w.field_configs.get_block()[0][::-1]):
            B = construct_propagator_matrix_ghf(system, BT2, c, conjt=True)
            for (idet, psi_i) in enumerate(psi_bp[iw].phi):
                # propagate each component of multi-determinant expansion
                psi_bp[iw].phi[idet] = B.dot(psi_bp[iw].phi[idet])
                if i != 0 and i % nstblz == 0:
                    # implicitly propagating the full GHF wavefunction
                    (psi_bp[iw].phi[idet], detR) = reortho(psi_i)
                    psi_bp[iw].weights[idet] *= detR.conjugate()
    return psi_bp


def back_propagate_single(phi_in, configs, weights, system, nstblz, BT2, store=False):
    r"""Perform back propagation for single walker.

    Parameters
    ---------
    phi_in : :class:`pie.walkers.Walker` object
        Walker.
    configs : :class:`numpy.ndarray`
        Auxilliary field configurations.
    weights : :class:`numpy.ndarray`
        Not used. For interface consistency.
    system : system object in general.
        Container for model input options.
    nstblz : int
        Number of steps between GS orthogonalisation.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    store : bool
        If true the the back propagated wavefunctions are stored along the back
        propagation path.

    Returns
    -------
    psi_store : list of :class:`pie.walker.Walker` objects
        Back propagated list of walkers.
    """
    nup = system.nup
    psi_store = []
    for (i, c) in enumerate(configs[::-1]):
        B = construct_propagator_matrix(system, BT2, c, conjt=True)
        phi_in[:, :nup] = B[0].dot(phi_in[:, :nup])
        phi_in[:, nup:] = B[1].dot(phi_in[:, nup:])
        if i != 0 and i % nstblz == 0:
            (phi_in[:, :nup], R) = reortho(phi_in[:, :nup])
            (phi_in[:, nup:], R) = reortho(phi_in[:, nup:])
        if store:
            psi_store.append(copy.deepcopy(phi_in))

    return psi_store


def back_propagate_single_ghf(phi, configs, weights, system, nstblz, BT2, store=False):
    r"""Perform back propagation for single walker.

    Parameters
    ---------
    phi : :class:`pie.walkers.MultiGHFWalker` object
        Walker.
    configs : :class:`numpy.ndarray`
        Auxilliary field configurations.
    weights : :class:`numpy.ndarray`
        Not used. For interface consistency.
    system : system object in general.
        Container for model input options.
    nstblz : int
        Number of steps between GS orthogonalisation.
    BT2 : :class:`numpy.ndarray`
        One body propagator.
    store : bool
        If true the the back propagated wavefunctions are stored along the back
        propagation path.

    Returns
    -------
    psi_store : list of :class:`pie.walker.Walker` objects
        Back propagated list of walkers.
    """
    nup = system.nup
    psi_store = []
    for (i, c) in enumerate(configs[::-1]):
        B = construct_propagator_matrix_ghf(system, BT2, c, conjt=True)
        for (idet, psi_i) in enumerate(phi):
            # propagate each component of multi-determinant expansion
            phi[idet] = B.dot(phi[idet])
            if i != 0 and i % nstblz == 0:
                # implicitly propagating the full GHF wavefunction
                (phi[idet], detR) = reortho(psi_i)
                weights[idet] *= detR.conjugate()
        if store:
            psi_store.append(copy.deepcopy(phi))

    return psi_store


def kinetic_kspace(phi, system, btk):
    """Apply the kinetic energy projector in kspace.

    May be faster for very large dilute lattices.

    Parameters
    ---------
    phi : :class:`pie.walkers.MultiGHFWalker` object
        Walker.
    system : system object in general.
        Container for model input options.
    B : :class:`numpy.ndarray`
        One body propagator.
    """
    s = system
    # Transform psi to kspace by fft-ing its columns.
    tup = fft_wavefunction(phi[:, : s.nup], s.nx, s.ny, s.nup, phi[:, : s.nup].shape)
    tdown = fft_wavefunction(
        phi[:, s.nup :], s.nx, s.ny, s.ndown, phi[:, s.nup :].shape
    )
    # Kinetic enery operator is diagonal in momentum space.
    # Note that multiplying by diagonal btk in this way is faster than using
    # einsum and way faster than using dot using an actual diagonal matrix.
    tup = (btk * tup.T).T
    tdown = (btk * tdown.T).T
    # Transform phi to kspace by fft-ing its columns.
    tup = ifft_wavefunction(tup, s.nx, s.ny, s.nup, tup.shape)
    tdown = ifft_wavefunction(tdown, s.nx, s.ny, s.ndown, tdown.shape)
    if phi.dtype == float:
        phi[:, : s.nup] = tup.astype(float)
        phi[:, s.nup :] = tdown.astype(float)
    else:
        phi[:, : s.nup] = tup
        phi[:, s.nup :] = tdown
