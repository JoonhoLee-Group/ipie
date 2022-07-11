import cmath
import math
import sys
import time

import numpy

from ipie.estimators.greens_function_batch import get_greens_function
from ipie.legacy.estimators.local_energy import local_energy
from ipie.legacy.propagation.generic import GenericContinuous
from ipie.legacy.propagation.hubbard import (HubbardContinuous,
                                             HubbardContinuousSpin)
from ipie.legacy.propagation.planewave import PlaneWave
from ipie.propagation.force_bias import construct_force_bias_batch
from ipie.propagation.operations import kinetic_real, kinetic_spin_real_batch
from ipie.propagation.overlap import get_calc_overlap
from ipie.utils.misc import is_cupy


class Continuous(object):
    """Propagation with continuous HS transformation."""

    def __init__(self, system, hamiltonian, trial, qmc, options={}, verbose=False):
        if verbose:
            print("# Parsing input options for propagation.Continuous.")
            print("# Using continuous Hubbard--Stratonovich transformations.")
        # Input options
        self.free_projection = options.get("free_projection", False)
        self.hybrid = options.get("hybrid", True)
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
        # Derived Attributes
        self.dt = qmc.dt
        self.sqrt_dt = qmc.dt**0.5
        self.isqrt_dt = 1j * self.sqrt_dt

        self.propagator = get_continuous_propagator(
            system, hamiltonian, trial, qmc, options=options, verbose=verbose
        )
        # self.calc_overlap = get_calc_overlap(trial)
        # self.compute_greens_function = get_greens_function(trial)

        assert qmc.batched == False or qmc.batched == None

        if self.hybrid:
            if verbose:
                print("# Using hybrid weight update.")
            self.update_weight = self.update_weight_hybrid
        else:
            if verbose:
                print("# Using local energy weight update.")
            self.update_weight = self.update_weight_local_energy
        # Constant core contribution modified by mean field shift.
        mf_core = self.propagator.mf_core

        self.log_mf_const_fac = -self.dt * mf_core.real

        # JOONHO - isn't this repeating the work? We will check this later; it's used by Hubbard Continuous
        self.propagator.construct_one_body_propagator(hamiltonian, qmc.dt)

        self.BT_BP = self.propagator.BH1
        self.nstblz = qmc.nstblz
        self.nfb_trig = 0
        self.nhe_trig = 0

        self.ebound = (2.0 / self.dt) ** 0.5

        if self.free_projection:
            if verbose:
                print("# Using free projection.")
            self.propagate_walker = self.propagate_walker_free
        else:
            if verbose:
                print("# Using phaseless approximation.")
            self.propagate_walker = self.propagate_walker_phaseless
        self.verbose = verbose

        self.tfbias = 0.0
        self.tovlp = 0.0
        self.tupdate = 0.0
        self.tupdate1 = 0.0
        self.tupdate2 = 0.0
        self.tupdate3 = 0.0
        self.tupdate4 = 0.0
        self.tupdate5 = 0.0
        self.tupdate6 = 0.0
        self.tupdate7 = 0.0
        self.tupdate8 = 0.0
        self.tupdate9 = 0.0
        self.tupdate10 = 0.0
        self.tupdate11 = 0.0
        self.tupdate12 = 0.0
        self.tupdate13 = 0.0
        self.tupdate14 = 0.0
        self.tupdate15 = 0.0
        self.tupdate16 = 0.0
        self.tgf = 0.0
        self.tvhs = 0.0
        self.tgemm = 0.0

    @property
    def mf_const_fac(self):
        return math.exp(self.log_mf_const_fac)

    @mf_const_fac.setter
    def mf_const_fac(self, value):
        self.mf_const_fac = value

    @mf_const_fac.deleter
    def mf_const_fac(self):
        del self.mf_const_fac

    def apply_exponential(self, phi, VHS, debug=False):
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
        if is_cupy(
            VHS
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy

            assert cupy.is_available()
            copy = cupy.copy
            copyto = cupy.copyto
            zeros = cupy.zeros
        else:
            copy = numpy.copy
            copyto = numpy.copyto
            zeros = numpy.zeros

        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)

        # Temporary array for matrix exponentiation.
        Temp = zeros(phi.shape, dtype=phi.dtype)

        copyto(Temp, phi)
        for n in range(1, self.exp_nmax + 1):
            Temp = VHS.dot(Temp) / n
            phi += Temp

        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))

        if is_cupy(
            VHS
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy

            cupy.cuda.stream.get_current_stream().synchronize()

        return phi

    def two_body_propagator(self, walker, system, hamiltonian, trial):
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
        if self.force_bias:
            start_time = time.time()
            xbar = self.propagator.construct_force_bias(hamiltonian, walker, trial)
            self.tfbias += time.time() - start_time

        absxbar = numpy.abs(xbar)
        idx_to_rescale = absxbar > 1.0
        nonzeros = absxbar > 1e-13
        xbar_rescaled = xbar.copy()
        xbar_rescaled[nonzeros] = xbar_rescaled[nonzeros] / absxbar[nonzeros]
        xbar = numpy.where(idx_to_rescale, xbar_rescaled, xbar)
        self.nfb_trig += numpy.sum(idx_to_rescale)

        xshifted = xi - xbar

        # Constant factor arising from force bias and mean field shift
        cmf = -self.sqrt_dt * xshifted.dot(self.propagator.mf_shift)
        # Constant factor arising from shifting the propability distribution.
        cfb = xi.dot(xbar) - 0.5 * xbar.dot(xbar)

        # Operator terms contributing to propagator.
        start_time = time.time()
        VHS = self.propagator.construct_VHS(hamiltonian, xshifted)
        self.tvhs += time.time() - start_time
        start_time = time.time()
        if len(VHS.shape) == 3:
            # 2.b Apply two-body
            self.apply_exponential(walker.phi[:, : system.nup], VHS[0])
            if system.ndown > 0:
                self.apply_exponential(walker.phi[:, system.nup :], VHS[1])
        else:
            # 2.b Apply two-body
            self.apply_exponential(walker.phi[:, : system.nup], VHS)
            if system.ndown > 0:
                self.apply_exponential(walker.phi[:, system.nup :], VHS)
        self.tgemm += time.time() - start_time

        return (cmf, cfb, xshifted)

    def propagate_walker_free(self, walker, system, hamiltonian, trial, eshift):
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
        # 1. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.propagator.BH1)
        # 2. Apply 2-body projector
        (cmf, cfb, xmxbar) = self.two_body_propagator(
            walker, system, hamiltonian, trial
        )
        # 3. Apply kinetic projector.
        kinetic_real(walker.phi, system, self.propagator.BH1)
        ovlp_new = walker.calc_overlap(trial)
        # Constant terms are included in the walker's weight.
        (magn, dtheta) = cmath.polar(cmath.exp(cmf + self.dt * eshift))
        walker.weight *= magn
        walker.phase *= cmath.exp(1j * dtheta)
        walker.ot = ovlp_new
        walker.ovlp = ovlp_new

    def apply_bound_hybrid(self, ehyb, eshift):
        # For initial steps until first estimator communication eshift will be
        # zero and hybrid energy can be incorrect. So just avoid capping for
        # first block until reasonable estimate of eshift can be computed.
        if abs(eshift) < 1e-10:
            return ehyb
        eoriginal = ehyb
        if ehyb.real > eshift.real + self.ebound:
            ehyb = eshift.real + self.ebound + 1j * ehyb.imag
            self.nhe_trig += 1
        elif ehyb.real < eshift.real - self.ebound:
            ehyb = eshift.real - self.ebound + 1j * ehyb.imag
            self.nhe_trig += 1
        return ehyb

    def apply_bound_local_energy(self, eloc, eshift):
        # For initial steps until first estimator communication eshift will be
        # zero and hybrid energy can be incorrect. So just avoid capping for
        # first block until reasonable estimate of eshift can be computed.
        if abs(eshift) < 1e-10:
            return eloc
        if eloc.real > eshift.real + self.ebound:
            eloc_bounded = eshift.real + self.ebound
            self.nhe_trig += 1
        elif eloc.real < eshift.real - self.ebound:
            eloc_bounded = eshift.real - self.ebound
            self.nhe_trig += 1
        else:
            eloc_bounded = eloc
        return eloc_bounded

    def propagate_walker_phaseless(self, walker, system, hamiltonian, trial, eshift):
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
        ovlp = walker.greens_function(trial)
        # 2. Update Slater matrix
        # 2.a Apply one-body
        kinetic_real(walker.phi, system, self.propagator.BH1)
        # 2.b Apply two-body
        (cmf, cfb, xmxbar) = self.two_body_propagator(
            walker, system, hamiltonian, trial
        )
        # 2.c Apply one-body
        kinetic_real(walker.phi, system, self.propagator.BH1)

        # Now apply phaseless approximation
        ovlp_new = walker.calc_overlap(trial)
        self.update_weight(
            system, hamiltonian, walker, trial, ovlp, ovlp_new, cfb, cmf, xmxbar, eshift
        )

    def update_weight_hybrid(
        self,
        system,
        hamiltonian,
        walker,
        trial,
        ovlp,
        ovlp_new,
        cfb,
        cmf,
        xmxbar,
        eshift,
    ):
        ovlp_ratio = ovlp_new / ovlp
        hybrid_energy = -(cmath.log(ovlp_ratio) + cfb + cmf) / self.dt
        hybrid_energy = self.apply_bound_hybrid(hybrid_energy, eshift)
        importance_function = cmath.exp(
            -self.dt * (0.5 * (hybrid_energy + walker.hybrid_energy) - eshift)
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
            walker.ot = ovlp_new
            walker.ovlp = ovlp_new
            if magn > 1e-16:
                wfac = numpy.array([importance_function / magn, cosine_fac])
            else:
                wfac = numpy.array([0, 0])
            if walker.field_configs is not None:
                walker.field_configs.update(xmxbar, wfac)
        else:
            walker.ot = ovlp_new
            walker.weight = 0.0

    def update_weight_local_energy(
        self,
        system,
        hamiltonian,
        walker,
        trial,
        ovlp,
        ovlp_new,
        cfb,
        cmf,
        xmxbar,
        eshift,
    ):
        ovlp_ratio = ovlp_new / ovlp
        eloc = local_energy(system, hamiltonian, walker, trial)[0]
        re_eloc = self.apply_bound_local_energy(eloc, eshift)
        magn = numpy.exp(-0.5 * self.dt * (re_eloc + walker.eloc - eshift).real)
        # for back propagation
        wfac_imag = numpy.exp(-0.5 * self.dt * (eloc + walker.eloc - eshift).imag)
        walker.eloc = eloc
        if not math.isinf(magn):
            # Determine cosine phase from Arg(<psi_T|B(x-\bar{x})|phi>/<psi_T|phi>)
            # Note this doesn't include exponential factor from shifting
            # propability distribution.
            dtheta = cmath.phase(ovlp_ratio)
            cosine_fac = max(0, math.cos(dtheta))
            walker.weight *= magn * cosine_fac
            walker.ot = ovlp_new
            if magn > 1e-16:
                wfac = numpy.array([wfac_imag, cosine_fac])
            else:
                wfac = numpy.array([0, 0])
            if walker.field_configs is not None:
                walker.field_configs.update(xmxbar, wfac)
        else:
            walker.ot = ot_new
            walker.weight = 0.0


def get_continuous_propagator(
    system, hamiltonian, trial, qmc, options={}, verbose=False
):
    """Wrapper to select propagator class.

    Parameters
    ----------
    options : dict
        Propagator input options.
    qmc : :class:`pie.qmc.QMCOpts` class
        Trial wavefunction input options.
    system : class
        System class.
    trial : class
        Trial wavefunction object.

    Returns
    -------
    propagator : class or None
        Propagator object.
    """
    if hamiltonian.name == "UEG":
        propagator = PlaneWave(
            system, hamiltonian, trial, qmc, options=options, verbose=verbose
        )
    elif hamiltonian.name == "Hubbard":
        charge = options.get("charge_decomposition", True)
        if charge:
            propagator = HubbardContinuous(
                hamiltonian, trial, qmc, options=options, verbose=verbose
            )
        else:
            propagator = HubbardContinuousSpin(
                hamiltonian, trial, qmc, options=options, verbose=verbose
            )
    elif hamiltonian.name == "Generic":
        propagator = GenericContinuous(
            system, hamiltonian, trial, qmc, options=options, verbose=verbose
        )
    else:
        propagator = None

    return propagator
