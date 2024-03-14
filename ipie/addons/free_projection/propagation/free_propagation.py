import time

import numpy

from ipie.propagation.phaseless_base import (
    construct_mean_field_shift,
    construct_one_body_propagator,
)
from ipie.propagation.phaseless_generic import PhaselessGeneric
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize


class FreePropagation(PhaselessGeneric):
    """fp-afqmc propagator"""

    def __init__(
        self, time_step: float, exp_nmax: int = 6, verbose: bool = False, ene_0: float = 0.0
    ) -> None:
        super().__init__(time_step, exp_nmax=exp_nmax, verbose=verbose)
        self.e_shift = ene_0  # unlike the dynamic shift in phaseless, this is a constant shift

    def build(self, hamiltonian, trial=None, walkers=None, mpi_handler=None, verbose=False):
        # dt/2 one-body propagator
        start = time.time()
        self.mf_shift = construct_mean_field_shift(hamiltonian, trial)
        if verbose:
            print(f"# Time to mean field shift: {time.time() - start} s")
            print(
                "# Absolute value of maximum component of mean field shift: "
                "{:13.8e}.".format(numpy.max(numpy.abs(self.mf_shift)))
            )
        # construct one-body propagator
        self.expH1 = construct_one_body_propagator(hamiltonian, self.mf_shift, self.dt)

        # # Allocate force bias (we don't need to do this here - it will be allocated when it is needed)
        self.vbias = None
        # self.vbias = numpy.zeros((walkers.nwalkers, hamiltonian.nfields),
        #                         dtype=numpy.complex128)
        self.e_shift_1 = -hamiltonian.ecore - xp.sum(self.mf_shift**2) / 2.0

    def propagate_walkers_two_body(self, walkers, hamiltonian, trial):
        # Normally distrubted auxiliary fields.
        xi = (
            xp.random.normal(0.0, 1.0, hamiltonian.nfields * walkers.nwalkers).reshape(
                walkers.nwalkers, hamiltonian.nfields
            )
            + 0.0j
        )

        # Constant factor arising from mean field shift
        cmf = xp.exp(-self.sqrt_dt * xp.einsum("wx,x->w", xi, self.mf_shift))
        # Constant factor arising from shifting the propability distribution.
        ceshift = xp.exp(self.dt * (self.e_shift + self.e_shift_1))
        xi = xi.T.copy()
        self.apply_VHS(walkers, hamiltonian, xi)

        return cmf, ceshift

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift):
        synchronize()
        start_time = time.time()
        ovlp = trial.calc_greens_function(walkers)
        synchronize()
        self.timer.tgf += time.time() - start_time

        # 2. Update Slater matrix
        # 2.a Apply one-body
        self.propagate_walkers_one_body(walkers)

        # 2.b Apply two-body
        (cmf, ceshift) = self.propagate_walkers_two_body(walkers, hamiltonian, trial)

        # 2.c Apply one-body
        self.propagate_walkers_one_body(walkers)

        start_time = time.time()
        ovlp_new = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_weight(walkers, ovlp, ovlp_new, ceshift, cmf, eshift)
        synchronize()
        self.timer.tupdate += time.time() - start_time

    def update_weight(self, walkers, ovlp, ovlp_new, cfb, cmf, eshift):
        # weights in fp keep track of the walker normalization
        magn, dtheta = xp.abs(cfb * cmf), xp.angle(cfb * cmf)
        walkers.weight *= magn
        walkers.phase *= xp.exp(1j * dtheta)
        walkers.ovlp = ovlp_new
