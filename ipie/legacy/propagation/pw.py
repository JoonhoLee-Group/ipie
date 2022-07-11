import cmath
import math
import sys

import numpy
import scipy

from ipie.estimators.utils import convolve, scipy_fftconvolve
from ipie.propagation.operations import kinetic_real


class PW(object):
    """Propagation with continuous HS transformation."""

    def __init__(self, system, trial, qmc, options={}, verbose=False):
        if verbose:
            print("# Parsing propagator input options.")
        # Input options
        self.free_projection = options.get("free_projection", False)
        if verbose:
            print("# Using phaseless approximation: %r" % (not self.free_projection))
        self.force_bias = options.get("force_bias", True)
        if self.free_projection:
            if verbose:
                print("# Setting force_bias to False with free projection.")
            self.force_bias = False
        else:
            if verbose:
                print("# Setting force bias to %r." % self.force_bias)
        self.exp_nmax = options.get("expansion_order", 6)
        if verbose:
            print("# expansion_order = {}".format(self.exp_nmax))
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j * self.sqrt_dt

        self.num_vplus = system.nfields // 2
        self.vbias = numpy.zeros(system.nfields, dtype=numpy.complex128)

        # Mean-field shift is zero for UEG.
        self.mf_shift = numpy.zeros(system.nfields, dtype=numpy.complex128)
        optimised = options.get("optimised", True)

        if optimised:
            self.construct_force_bias = self.construct_force_bias_fft
        else:
            print("# Slow routines not available. Please Implement.")
            sys.exit()

        # Input options
        if verbose:
            print("# Finished setting up plane wave propagator.")

        # Constant core contribution modified by mean field shift.
        self.mf_core = 0.0  # Hard-coded to be zero for now
        self.mf_const_fac = math.exp(-self.dt * self.mf_core.real)
        self.construct_one_body_propagator(system, qmc.dt)
        self.BT_BP = self.BH1
        self.nstblz = qmc.nstblz
        self.nfb_trig = 0
        self.nhe_trig = 0

        self.ebound = (2.0 / self.dt) ** 0.5
        self.mean_local_energy = 0

        if self.free_projection:
            if verbose:
                print("# Using free projection.")
            self.propagate_walker = self.propagate_walker_free
        else:
            if verbose:
                print("# Using phaseless approximation.")
            self.propagate_walker = self.propagate_walker_phaseless
        self.verbose = verbose

    def apply_two_body_propagator(self, walker, system, trial):
        """It appliese the two-body propagator
        Parameters
        ----------
        walker :
            walker class
        system :
            system class
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
        xi = numpy.random.normal(0.0, 1.0, system.nfields)

        # Optimal force bias.
        xbar = numpy.zeros(system.nfields)
        if self.force_bias:
            xbar = self.construct_force_bias(system, walker, trial)

        for i in range(system.nfields):
            if numpy.absolute(xbar[i]) > 1.0:
                if self.nfb_trig < 1:
                    if self.verbose:
                        pass
                self.nfb_trig += 1
                xbar[i] /= numpy.absolute(xbar[i])

        xshifted = xi - xbar

        # Constant factor arising from force bias and mean field shift
        cmf = -self.sqrt_dt * xshifted.dot(self.mf_shift)
        # Constant factor arising from shifting the propability distribution.
        cfb = xi.dot(xbar) - 0.5 * xbar.dot(xbar)

        # two-body propagator starts
        nocc = [system.nup, system.ndown]
        ngrid = numpy.prod(system.mesh)
        nqgrid = numpy.prod(system.qmesh)

        vqfactor = (
            self.sqrt_dt * numpy.sqrt(1.0 / (4.0 * system.vol)) * system.sqrtvqvec
        )

        x_plus_cube = numpy.zeros(nqgrid, dtype=numpy.complex128)
        x_plus_cube[system.qmap] = xshifted[: self.num_vplus] * vqfactor

        x_minus_cube = numpy.zeros(nqgrid, dtype=numpy.complex128)
        x_minus_cube[system.qmap] = xshifted[self.num_vplus :] * vqfactor

        expVphi = walker.phi.copy()
        Vphi_grid = numpy.zeros((ngrid, nocc[0] + nocc[1]), dtype=numpy.complex128)
        Vphi_grid[system.gmap, :] = walker.phi.copy()
        for n in range(1, self.exp_nmax + 1):
            for i in range(nocc[0] + nocc[1]):
                # \sum_Q X(Q) * phi(G-Q)
                l1 = scipy_fftconvolve(
                    x_plus_cube,
                    Vphi_grid[:, i],
                    mesh1=system.qmesh,
                    mesh2=system.mesh,
                    mode="valid",
                ).ravel()
                # \sum_Q X(Q) * phi(G+Q)
                l2 = scipy_fftconvolve(
                    x_plus_cube,
                    Vphi_grid[:, i][::-1],
                    mesh1=system.qmesh,
                    mesh2=system.mesh,
                    mode="valid",
                ).ravel()[::-1]
                # \sum_Q X(Q) * phi(G-Q)
                l3 = scipy_fftconvolve(
                    x_minus_cube,
                    Vphi_grid[:, i],
                    mesh1=system.qmesh,
                    mesh2=system.mesh,
                    mode="valid",
                ).ravel()
                # \sum_Q X(Q) * phi(G+Q)
                l4 = scipy_fftconvolve(
                    x_minus_cube,
                    Vphi_grid[:, i][::-1],
                    mesh1=system.qmesh,
                    mesh2=system.mesh,
                    mode="valid",
                ).ravel()[::-1]

                Vphi_grid[:, i] = (1.0j * (l1 + l2) - (l3 - l4)) / float(n)

            tmp = Vphi_grid.copy()
            Vphi_grid[:, :] = 0.0
            Vphi_grid[system.gmap, :] = tmp[system.gmap, :]

            expVphi += Vphi_grid[system.gmap, :]

        walker.phi = expVphi.copy()

        return (cmf, cfb, xshifted)

    def propagate_walker_free(self, walker, system, trial, eshift):
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
        # 1. Apply one-body
        kinetic_real(walker.phi, system, self.BH1, system.diagH1)
        # 2. Apply two-body propagator.
        (cmf, cfb, xmxbar) = self.apply_two_body_propagator(walker, system, trial)
        # 3. Apply one-body
        kinetic_real(walker.phi, system, self.BH1, system.diagH1)

        walker.inverse_overlap(trial)
        walker.ot = walker.calc_otrial(trial)
        walker.greens_function(trial)
        # Constant terms are included in the walker's weight.
        (magn, dtheta) = cmath.polar(cmath.exp(cmf + self.dt * eshift))
        walker.weight *= magn
        walker.phase *= cmath.exp(1j * dtheta)

    def apply_bound(self, ehyb, eshift):
        if ehyb.real > eshift.real + self.ebound:
            ehyb = eshift.real + self.ebound + 1j * ehyb.imag
            self.nhe_trig += 1
        elif ehyb.real < eshift.real - self.ebound:
            ehyb = eshift.real - self.ebound + 1j * ehyb.imag
            self.nhe_trig += 1
        return ehyb

    def propagate_walker_phaseless(self, walker, system, trial, eshift):
        """Phaseless propagator
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
        # Update Slater matrix
        # 1. Apply one-body
        kinetic_real(walker.phi, system, self.BH1, system.diagH1)
        # 2. Apply two-body propagator.
        (cmf, cfb, xmxbar) = self.apply_two_body_propagator(walker, system, trial)

        # 3. Apply one-body
        kinetic_real(walker.phi, system, self.BH1, system.diagH1)

        # Now apply phaseless approximation
        walker.inverse_overlap(trial)
        walker.greens_function(trial)
        ot_new = walker.calc_otrial(trial)
        ovlp_ratio = ot_new / walker.ot
        hybrid_energy = -(cmath.log(ovlp_ratio) + cfb + cmf) / self.dt
        hybrid_energy = self.apply_bound(hybrid_energy, eshift)
        importance_function = (
            # self.mf_const_fac * No need to include constant factor.
            cmath.exp(
                -self.dt * (0.5 * (hybrid_energy + walker.hybrid_energy) - eshift)
            )
        )
        # splitting w_alpha = |I(x,\bar{x},|phi_alpha>)| e^{i theta_alpha}
        (magn, phase) = cmath.polar(importance_function)
        walker.hybrid_energy = hybrid_energy

        if not math.isinf(magn):
            # Determine cosine phase from Arg(<psi_T|B(x-\bar{x})|phi>/<psi_T|phi>)
            # Note this doesn't include exponential factor from shifting
            # propability distribution.
            dtheta = (-self.dt * hybrid_energy - cfb).imag
            cosine_fac = max(0, math.cos(dtheta))
            walker.weight *= magn * cosine_fac
            walker.ot = ot_new
            if magn > 1e-16:
                wfac = numpy.array([importance_function / magn, cosine_fac])
            else:
                wfac = numpy.array([0, 0])
            try:
                walker.field_configs.update(xmxbar, wfac)
            except AttributeError:
                pass
        else:
            walker.ot = ot_new
            walker.weight = 0.0

    def construct_one_body_propagator(self, system, dt):
        """Construct the one-body propagator Exp(-dt/2 H0)
        Parameters
        ----------
        system :
            system class
        dt : float
            time-step
        Returns
        -------
        self.BH1 : numpy array
            Exp(-dt/2 H0)
        """
        H1 = system.h1e_mod
        # No spin dependence for the moment.
        if system.diagH1:
            self.BH1 = numpy.array(
                [
                    numpy.diag(numpy.exp(-0.5 * dt * numpy.diag(H1[0]))),
                    numpy.diag(numpy.exp(-0.5 * dt * numpy.diag(H1[1]))),
                ]
            )
        else:
            self.BH1 = numpy.array(
                [
                    scipy.linalg.expm(-0.5 * dt * H1[0]),
                    scipy.linalg.expm(-0.5 * dt * H1[1]),
                ]
            )

    def construct_force_bias_fft(self, system, walker, trial):
        """Compute the force bias term as in Eq.(33) of DOI:10.1002/wcms.1364
        Parameters
        ----------
        system :
            system class
        G : numpy array
            Green's function
        Returns
        -------
        force bias : numpy array
            -sqrt(dt) * vbias
        """
        G = walker.G
        Ghalf = walker.Gmod

        nocc = [system.nup, system.ndown]
        ngrid = numpy.prod(system.mesh)
        factor = numpy.sqrt(1.0 / (4.0 * system.vol))

        CTdagger = numpy.array(
            [
                numpy.array(
                    trial.psi[:, 0 : system.nup], dtype=numpy.complex128
                ).T.conj(),
                numpy.array(
                    trial.psi[:, system.nup :], dtype=numpy.complex128
                ).T.conj(),
            ]
        )

        self.vbias[:] = 0.0 + 0.0j

        for s in [0, 1]:
            for i in range(nocc[s]):
                Gh_i = Ghalf[s][i, :]
                CTdagger_i = CTdagger[s][i, :]

                Gh_i_cube = numpy.zeros(ngrid, dtype=numpy.complex128)
                CTdagger_i_cube = numpy.zeros(ngrid, dtype=numpy.complex128)

                Gh_i_cube[system.gmap] = Gh_i
                CTdagger_i_cube[system.gmap] = CTdagger_i

                # \sum_G CT(G+Q) theta(G)
                lQ_i1 = numpy.flip(
                    convolve(Gh_i_cube, numpy.flip(CTdagger_i_cube), system.mesh)
                )[system.qmap]
                # \sum_G CT(G) theta(G+Q)
                lQ_i2 = numpy.flip(
                    convolve(CTdagger_i_cube, numpy.flip(Gh_i_cube), system.mesh)
                )[system.qmap]

                self.vbias[: self.num_vplus] += (lQ_i1 + lQ_i2) * 1.0j
                self.vbias[self.num_vplus :] += -lQ_i1 + lQ_i2

        self.vbias[: self.num_vplus] *= factor * system.sqrtvqvec
        self.vbias[self.num_vplus :] *= factor * system.sqrtvqvec

        return -self.sqrt_dt * self.vbias


def unit_test():
    import cProfile

    from ipie.legacy.systems.ueg import UEG
    from ipie.legacy.trial_wavefunction.free_electron import FreeElectron
    from ipie.legacy.trial_wavefunction.hartree_fock import HartreeFock
    from ipie.legacy.walkers.single_det import SingleDetWalker
    from ipie.propagation.continuous import Continuous
    from ipie.qmc.options import QMCOpts
    from ipie.systems.pw_fft import PW_FFT

    inputs = {
        "nup": 1,
        "ndown": 1,
        "rs": 1.0,
        "ecut": 20.0,
        "dt": 0.05,
        "nwalkers": 1,
        "expansion_order": 6,
    }

    numpy.random.seed(7)

    system = PW_FFT(inputs, True)
    qmc = QMCOpts(inputs, system, True)
    trial = HartreeFock(system, False, inputs, True)
    rpsi = numpy.random.rand(system.nbasis, system.nup + system.ndown)
    zpsi = numpy.random.rand(system.nbasis, system.nup + system.ndown)
    trial.psi = rpsi + 1.0j * zpsi

    propagator = PW(system, trial, qmc, inputs, verbose=True)

    walker = SingleDetWalker({}, system, trial, index=0)
    walker.greens_function(trial)

    rphi = numpy.random.rand(system.nbasis, system.nup + system.ndown)
    zphi = numpy.random.rand(system.nbasis, system.nup + system.ndown)
    walker.phi = rphi + 1.0j * zphi

    eshift = 0.0 + 0.0j

    # print(walker.phi)
    pr = cProfile.Profile()
    pr.enable()
    propagator.propagate_walker_phaseless(
        walker=walker, system=system, trial=trial, eshift=eshift
    )
    propagator.propagate_walker_phaseless(
        walker=walker, system=system, trial=trial, eshift=eshift
    )
    propagator.propagate_walker_phaseless(
        walker=walker, system=system, trial=trial, eshift=eshift
    )
    propagator.propagate_walker_phaseless(
        walker=walker, system=system, trial=trial, eshift=eshift
    )
    propagator.propagate_walker_phaseless(
        walker=walker, system=system, trial=trial, eshift=eshift
    )
    pr.disable()
    pr.print_stats(sort="tottime")
    # print(walker.phi)
    # phi_ref = walker.phi.copy()

    sort_basis = numpy.argsort(numpy.diag(system.H1[0]), kind="mergesort")
    numpy.random.seed(7)

    system = UEG(inputs, True)
    qmc = QMCOpts(inputs, system, True)
    trial = HartreeFock(system, False, inputs, True)
    rpsi = numpy.random.rand(system.nbasis, system.nup + system.ndown)
    zpsi = numpy.random.rand(system.nbasis, system.nup + system.ndown)
    trial.psi = rpsi + 1.0j * zpsi
    trial.psi[:, :] = trial.psi[sort_basis, :]

    propagator = Continuous(system, trial, qmc, inputs, verbose=True)

    walker = SingleDetWalker({}, system, trial, index=0)
    walker.greens_function(trial)

    rphi = numpy.random.rand(system.nbasis, system.nup + system.ndown)
    zphi = numpy.random.rand(system.nbasis, system.nup + system.ndown)
    walker.phi = rphi + 1.0j * zphi
    walker.phi[:, :] = walker.phi[sort_basis, :]

    # print(walker.phi)

    eshift = 0.0 + 0.0j

    pr = cProfile.Profile()
    pr.enable()
    propagator.propagate_walker_phaseless(
        walker=walker, system=system, trial=trial, eshift=eshift
    )
    propagator.propagate_walker_phaseless(
        walker=walker, system=system, trial=trial, eshift=eshift
    )
    propagator.propagate_walker_phaseless(
        walker=walker, system=system, trial=trial, eshift=eshift
    )
    propagator.propagate_walker_phaseless(
        walker=walker, system=system, trial=trial, eshift=eshift
    )
    propagator.propagate_walker_phaseless(
        walker=walker, system=system, trial=trial, eshift=eshift
    )
    pr.disable()
    pr.print_stats(sort="tottime")
    # print(walker.phi)
    # print(phi_ref)]


if __name__ == "__main__":
    unit_test()
