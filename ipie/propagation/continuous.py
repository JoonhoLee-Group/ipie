import cmath
import math
import sys
import time

import numpy

from ipie.estimators.greens_function_batch import compute_greens_function
from ipie.legacy.estimators.local_energy import local_energy
from ipie.propagation.force_bias import construct_force_bias_batch
from ipie.propagation.generic import GenericContinuous
from ipie.propagation.operations import kinetic_real, kinetic_spin_real_batch
from ipie.propagation.overlap import get_calc_overlap
from ipie.utils.misc import is_cupy
from ipie.qmc.utils import gpu_synchronize


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

        assert hamiltonian.name == "Generic"

        self.propagator = GenericContinuous(
            system, hamiltonian, trial, qmc, options=options, verbose=verbose
        )

        self.calc_overlap = get_calc_overlap(trial)
        # self.compute_greens_function = get_greens_function(trial)

        assert self.hybrid
        if verbose:
            print("# Using hybrid weight update.")
        self.update_weight_batch = self.update_weight_hybrid_batch

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

        assert not self.free_projection
        if verbose:
            print("# Using phaseless approximation.")
        self.propagate_walker_batch = self.propagate_walker_phaseless_batch

        self.verbose = verbose

        self.tfbias = 0.0
        self.tovlp = 0.0
        self.tupdate = 0.0
        self.tgf = 0.0
        self.tvhs = 0.0
        self.tgemm = 0.0

    def cast_to_cupy(self, verbose=False):
        import cupy

        size = (
            self.propagator.mf_shift.size
            + self.propagator.vbias_batch.size
            + self.propagator.BH1.size
        )
        size += 1  # for ebound
        if verbose:
            expected_bytes = size * 16.0  # assuming complex128
            print(
                "# propagators.Continuous: expected to allocate {:4.3f} GB".format(
                    expected_bytes / 1024**3
                )
            )
        self.propagator.mf_shift = cupy.asarray(self.propagator.mf_shift)
        self.propagator.vbias_batch = cupy.asarray(self.propagator.vbias_batch)
        self.propagator.BH1 = cupy.asarray(self.propagator.BH1)
        self.ebound = cupy.asarray(self.ebound)
        free_bytes, total_bytes = cupy.cuda.Device().mem_info
        used_bytes = total_bytes - free_bytes
        if verbose:
            print(
                "# propagators.Continuous: using {:4.3f} GB out of {:4.3f} GB memory on GPU".format(
                    used_bytes / 1024**3, total_bytes / 1024**3
                )
            )

    @property
    def mf_const_fac(self):
        return math.exp(self.log_mf_const_fac)

    @mf_const_fac.setter
    def mf_const_fac(self, value):
        self.mf_const_fac = value

    @mf_const_fac.deleter
    def mf_const_fac(self):
        del self.mf_const_fac

    def apply_exponential_batch(self, phi, VHS, debug=False):
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
            einsum = cupy.einsum
            gpu = True
        else:
            copy = numpy.copy
            copyto = numpy.copyto
            zeros = numpy.zeros
            einsum = numpy.einsum
            gpu = False

        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)

        # Temporary array for matrix exponentiation.
        Temp = zeros(phi.shape, dtype=phi.dtype)

        copyto(Temp, phi)
        if is_cupy(VHS):
            for n in range(1, self.exp_nmax + 1):
                Temp = einsum("wik,wkj->wij", VHS, Temp, optimize=True) / n
                phi += Temp
        else:
            for iw in range(phi.shape[0]):
                for n in range(1, self.exp_nmax + 1):
                    Temp[iw] = VHS[iw].dot(Temp[iw]) / n
                    phi[iw] += Temp[iw]
        
        gpu_synchronize(gpu)

        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))
        return phi

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
            gpu = True
        else:
            copy = numpy.copy
            copyto = numpy.copyto
            zeros = numpy.zeros
            gpu = False

        if debug:
            copy = numpy.copy(phi)
            c2 = scipy.linalg.expm(VHS).dot(copy)

        # Temporary array for matrix exponentiation.
        Temp = zeros(phi.shape, dtype=phi.dtype)

        copyto(Temp, phi)
        for n in range(1, self.exp_nmax + 1):
            Temp = VHS.dot(Temp) / n
            phi += Temp

        gpu_synchronize(gpu)

        if debug:
            print("DIFF: {: 10.8e}".format((c2 - phi).sum() / c2.size))
        return phi

    def apply_bound_hybrid_batch(
        self, ehyb, eshift
    ):  # shift is a number but ehyb is not
        if is_cupy(
            ehyb
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy

            assert cupy.is_available()
            clip = cupy.clip
            gpu = True
        else:
            clip = numpy.clip
            gpu = False
        # For initial steps until first estimator communication eshift will be
        # zero and hybrid energy can be incorrect. So just avoid capping for
        # first block until reasonable estimate of eshift can be computed.
        if abs(eshift) < 1e-10:
            return ehyb
        emax = eshift.real + self.ebound
        emin = eshift.real - self.ebound
        clip(ehyb.real, a_min=emin, a_max=emax, out=ehyb.real)  # in-place clipping
        gpu_synchronize(gpu)
        return ehyb

    def two_body_propagator_batch(self, walker_batch, system, hamiltonian, trial):
        """It applies the two-body propagator to a batch of walkers
        Parameters
        ----------
        walker batch :
            walker_batch class
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
        if is_cupy(
            trial.psi
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy

            assert cupy.is_available()
            einsum = cupy.einsum
            abs = cupy.abs
            zeros = cupy.zeros
            normal = cupy.random.normal
            where = cupy.where
            sum = cupy.sum
            gpu = True
        else:
            einsum = numpy.einsum
            abs = numpy.abs
            zeros = numpy.zeros
            normal = numpy.random.normal
            where = numpy.where
            sum = numpy.sum
            gpu = False

        # Optimal force bias.
        xbar = zeros((walker_batch.nwalkers, hamiltonian.nfields))
        if self.force_bias:
            start_time = time.time()
            self.propagator.vbias_batch = construct_force_bias_batch(
                hamiltonian, walker_batch, trial, walker_batch.mpi_handler
            )
            xbar = -self.propagator.sqrt_dt * (
                1j * self.propagator.vbias_batch - self.propagator.mf_shift
            )
            gpu_synchronize(gpu)
            self.tfbias += time.time() - start_time

        absxbar = abs(xbar)
        idx_to_rescale = absxbar > 1.0
        nonzeros = absxbar > 1e-13
        xbar_rescaled = xbar.copy()
        xbar_rescaled[nonzeros] = xbar_rescaled[nonzeros] / absxbar[nonzeros]
        xbar = where(idx_to_rescale, xbar_rescaled, xbar)

        self.nfb_trig += sum(idx_to_rescale)

        # Normally distrubted auxiliary fields.
        xi = normal(0.0, 1.0, hamiltonian.nfields * walker_batch.nwalkers).reshape(
            walker_batch.nwalkers, hamiltonian.nfields
        )
        xshifted = xi - xbar

        # Constant factor arising from force bias and mean field shift
        cmf = -self.sqrt_dt * einsum("wx,x->w", xshifted, self.propagator.mf_shift)
        # Constant factor arising from shifting the propability distribution.
        cfb = einsum("wx,wx->w", xi, xbar) - 0.5 * einsum("wx,wx->w", xbar, xbar)

        # Operator terms contributing to propagator.
        start_time = time.time()
        if hamiltonian.chunked:
            VHS = self.propagator.construct_VHS_batch_chunked(
                hamiltonian, xshifted.T.copy(), walker_batch.mpi_handler
            )
        else:
            VHS = self.propagator.construct_VHS_batch(hamiltonian, xshifted.T.copy())
        gpu_synchronize(gpu)
        self.tvhs += time.time() - start_time
        assert len(VHS.shape) == 3
        start_time = time.time()
        if is_cupy(trial.psi):
            walker_batch.phia = self.apply_exponential_batch(walker_batch.phia, VHS)
            if walker_batch.ndown > 0 and not walker_batch.rhf:
                walker_batch.phib = self.apply_exponential_batch(walker_batch.phib, VHS)
        else:
            for iw in range(walker_batch.nwalkers):
                # 2.b Apply two-body
                walker_batch.phia[iw] = self.apply_exponential(
                    walker_batch.phia[iw], VHS[iw]
                )
                if walker_batch.ndown > 0 and not walker_batch.rhf:
                    walker_batch.phib[iw] = self.apply_exponential(
                        walker_batch.phib[iw], VHS[iw]
                    )
        gpu_synchronize(gpu)
        self.tgemm += time.time() - start_time

        return (cmf, cfb, xshifted)

    def propagate_walker_phaseless_batch(
        self, walker_batch, system, hamiltonian, trial, eshift
    ):
        """Phaseless propagator
        Parameters
        ----------
        walker :
            walker class
        system :
            system class
        trial :
            trial wavefunction class
        eshift :
            constant energy shift
        Returns
        -------
        """
        if is_cupy(
            walker_batch.phia
        ):
            gpu = True
        else:
            gpu = False

        gpu_synchronize(gpu)
        start_time = time.time()
        ovlp = compute_greens_function(walker_batch, trial)
        gpu_synchronize(gpu)
        self.tgf += time.time() - start_time

        # 2. Update Slater matrix
        # 2.a Apply one-body
        start_time = time.time()
        walker_batch.phia = kinetic_spin_real_batch(
            walker_batch.phia, self.propagator.BH1[0]
        )
        if walker_batch.ndown > 0 and not walker_batch.rhf:
            walker_batch.phib = kinetic_spin_real_batch(
                walker_batch.phib, self.propagator.BH1[1]
            )
        gpu_synchronize(gpu)
        self.tgemm += time.time() - start_time

        # 2.b Apply two-body
        (cmf, cfb, xmxbar) = self.two_body_propagator_batch(
            walker_batch, system, hamiltonian, trial
        )
        gpu_synchronize(gpu)

        # 2.c Apply one-body
        start_time = time.time()
        walker_batch.phia = kinetic_spin_real_batch(
            walker_batch.phia, self.propagator.BH1[0]
        )
        if walker_batch.ndown > 0 and not walker_batch.rhf:
            walker_batch.phib = kinetic_spin_real_batch(
                walker_batch.phib, self.propagator.BH1[1]
            )
        gpu_synchronize(gpu)
        self.tgemm += time.time() - start_time

        # Now apply phaseless approximation
        start_time = time.time()
        ovlp_new = self.calc_overlap(walker_batch, trial)
        gpu_synchronize(gpu)
        self.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_weight_batch(
            system,
            hamiltonian,
            walker_batch,
            trial,
            ovlp,
            ovlp_new,
            cfb,
            cmf,
            xmxbar,
            eshift,
        )
        gpu_synchronize(gpu)
        self.tupdate += time.time() - start_time

    def update_weight_hybrid_batch(
        self,
        system,
        hamiltonian,
        walker_batch,
        trial,
        ovlp,
        ovlp_new,
        cfb,
        cmf,
        xmxbar,
        eshift,
    ):
        if is_cupy(
            trial.psi
        ):  # if even one array is a cupy array we should assume the rest is done with cupy
            import cupy

            assert cupy.is_available()
            log = cupy.log
            exp = cupy.exp
            isinf = cupy.isinf
            isfinite = cupy.isfinite
            cos = cupy.cos
            array = cupy.array
            abs = cupy.abs
            angle = cupy.angle
            clip = cupy.clip
        else:
            log = numpy.log
            exp = numpy.exp
            isinf = numpy.isinf
            isfinite = numpy.isfinite
            cos = numpy.cos
            array = numpy.array
            abs = numpy.abs
            angle = numpy.angle
            clip = numpy.clip

        ovlp_ratio = ovlp_new / ovlp
        hybrid_energy = -(log(ovlp_ratio) + cfb + cmf) / self.dt
        hybrid_energy = self.apply_bound_hybrid_batch(hybrid_energy, eshift)
        importance_function = exp(
            -self.dt * (0.5 * (hybrid_energy + walker_batch.hybrid_energy) - eshift)
        )
        # splitting w_alpha = |I(x,\bar{x},|phi_alpha>)| e^{i theta_alpha}
        magn = abs(importance_function)
        phase = angle(importance_function)
        # (magn, phase) = cmath.polar(importance_function)
        walker_batch.hybrid_energy = hybrid_energy

        tosurvive = isfinite(magn)

        # disabling this because it seems unnecessary
        # tobeinstantlykilled = isinf(magn)
        # magn[tobeinstantlykilled] = 0.0

        dtheta = (-self.dt * hybrid_energy - cfb).imag
        cosine_fac = cos(dtheta)

        clip(cosine_fac, a_min=0.0, a_max=None, out=cosine_fac)  # in-place clipping
        walker_batch.weight = walker_batch.weight * magn * cosine_fac
        walker_batch.ovlp = ovlp_new

        #       TODO: make it a proper walker batching algorithm when we add back propagation
        if walker_batch.field_configs is not None:
            for iw in range(walker_batch.nwalkers):
                if tosurvive[iw]:
                    if magn > 1e-16:
                        wfac = array(
                            [importance_function[iw] / magn[iw], cosine_fac[iw]]
                        )
                    else:
                        wfac = array([0, 0])
                    walker_batch.field_configs[iw].update(xmxbar, wfac)
