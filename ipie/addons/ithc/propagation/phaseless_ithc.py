# Author: Maxine Luo <man.luo@mpq.mpg.de>
#         Victor Chen <victor.chen@tum.de>
#
from ipie.propagation.phaseless_base import PhaselessBase
from ipie.propagation.phaseless_base import construct_mean_field_shift
from ipie.addons.ithc.propagation.operations import apply_isometry

# from ipie.propagation.operations import propagate_one_body

# from ipie.estimators.energy import *
from ipie.addons.ithc.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.addons.ithc.trial_wavefunction.single_det import SingleDet
from typing import Union

# import math
# import copy
import time
import scipy.linalg

# from ipie.hamiltonians.generic_base import GenericBase
from ipie.addons.ithc.hamiltonians.generic_ithc import GenericITHC

# from ipie.estimators.local_energy_ithc import local_energy_single_det_uhf_ithc,compute_pe_batched,greens_function_ithc
import numpy

# from ipie.systems.generic import Generic

import plum

# from ipie.config import config

# from ipie.propagation.AFQMC_func import *
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.utils.misc import is_cupy


def construct_mean_field_shift(hamiltonian: GenericITHC, trial: SingleDet):
    r"""Compute mean field shift.

    .. math::

        \bar{v}_n = \sum_{ik\sigma} v_{(ik),n} G_{ik\sigma}

    """
    isometry = hamiltonian.isometry
    M = numpy.shape(isometry)[1]
    n = numpy.einsum("ja,ia,wij->wa", isometry, isometry, trial.G, optimize=True)
    n = numpy.reshape(n, (2 * M))
    mf_shift = hamiltonian.v @ n

    return mf_shift  # dimension (n_fields,)


def calc_noper_extended(psi0, psi0_extended, phi_batch, phi_extended_batch):
    """Applies operators in the extended basis set"""
    assert psi0.shape == phi_batch.shape[1:], "dimenstion dismatch!"
    nwalkers, nbasis, _ = phi_extended_batch.shape

    if is_cupy(psi0):
        O = xp.einsum("wmi,mj->wij", phi_batch, psi0.conj(), optimize=True)
        inv_O = xp.linalg.inv(O)
        n_batch = xp.einsum(
            "ni,wij,wnj->wn", psi0_extended.conj(), inv_O, phi_extended_batch, optimize=True
        )

    else:
        n_batch = xp.zeros(shape=(nwalkers, nbasis), dtype=xp.complex128)
        for iw in range(nwalkers):
            inv_O = xp.linalg.inv(xp.dot(phi_batch[iw].T, psi0.conj()))
            GHalf = xp.dot(inv_O, phi_extended_batch[iw].T)
            n_batch[iw] = xp.einsum("ni,in->n", psi0_extended.conj(), GHalf)

    return n_batch


def apply_diag_oper(phi, BV):

    phi *= BV[:, :, None]

    return


class PhaselessITHC(PhaselessBase):
    """A class for continuous HS transform with extended basis set ithc-AFQMC propagators."""

    def __init__(self, time_step, ebound_const=2.0, fbbound=1.0, verbose=False):
        super().__init__(time_step, ebound_const=ebound_const, fbbound=fbbound, verbose=verbose)

    def build(
        self,
        hamiltonian: GenericITHC,
        trial=None,
        walkers=None,
        mpi_handler=None,
        verbose=False,
        mean_field_shift=True,
    ):
        # dt/2 one-body propagator
        start = time.time()

        # Set up Isometry, one-body, eri and W from Hamiltonian class
        self.isometry = xp.asarray(hamiltonian.isometry)
        hamiltonian.isometry_test()
        self.W = xp.asarray(hamiltonian.W)
        self.v = xp.asarray(hamiltonian.v)  # n_fields * shape
        self.propagate_walkers_two_body = self.propagate_walkers_two_body_first_order

        if mean_field_shift:
            self.mf_shift = construct_mean_field_shift(hamiltonian, trial)
            tmp = xp.einsum("nm,n->m", self.v, self.mf_shift, optimize=True)
            tmp = tmp.reshape(2, self.isometry.shape[1])
            shift = xp.einsum("ia,ja,sa-> sij", self.isometry, self.isometry, tmp)
        else:
            self.mf_shift = xp.zeros(self.v.shape[0])
            shift = xp.zeros_like(hamiltonian.H1)

        if hasattr(shift, "get"):
            H1_numpy = hamiltonian.H1 - shift.get()
        else:
            H1_numpy = hamiltonian.H1 - shift

        self.expH1 = xp.array(
            [
                scipy.linalg.expm(-0.5 * self.dt * H1_numpy[0]),
                scipy.linalg.expm(-0.5 * self.dt * H1_numpy[1]),
            ]
        )

        # # Allocate force bias (we don't need to do this here - it will be allocated when it is needed)
        self.vbias = None
        self.timer.tgemm += time.time() - start
        return

    def construct_VHS(self, vbar):
        n_walkers = vbar.shape[0]
        n_fields = self.v.shape[0]
        n_extended = self.isometry.shape[1]
        assert self.v.shape[1] == n_extended * 2

        xbar = -self.sqrt_dt * (vbar - self.mf_shift)

        x = xp.random.normal(0.0, 1.0, (n_walkers, n_fields))
        xshifted = x - xbar
        T = self.sqrt_dt * xshifted @ self.v

        cfb_log = xp.einsum("wx,wx->w", x, xbar) - 0.5 * xp.einsum("wx,wx->w", xbar, xbar)
        cmf_log = -self.sqrt_dt * xp.einsum("wx,x->w", xshifted, self.mf_shift)

        synchronize()

        return T[:, :n_extended], T[:, n_extended:], cfb_log, cmf_log

    @plum.dispatch
    def apply_VHS(
        self, walkers: Union[UHFWalkers, GHFWalkers], hamiltonian: GenericITHC, xshifted: xp.ndarray
    ):
        pass

    def propagate_walkers_two_body_first_order(
        self, walkers, hamiltonian: GenericITHC, trial, half_rotate=True
    ):  # applies discrete HS transform and updates weights consider spin
        # force bias

        start_time = time.time()
        phia_transformed = apply_isometry(
            walkers.phia, self.isometry, reverse=True
        )  # first transform into extended basis set
        phib_transformed = apply_isometry(walkers.phib, self.isometry, reverse=True)
        self.timer.tgemm += time.time() - start_time

        start_time = time.time()
        na = calc_noper_extended(
            trial.psi0a, trial._psi0a_transformed, walkers.phia, phia_transformed
        )  # calculate greens function in extended basis
        nb = calc_noper_extended(
            trial.psi0b, trial._psi0b_transformed, walkers.phib, phib_transformed
        )
        na = self.apply_bound_force_bias(na, 1.0)
        nb = self.apply_bound_force_bias(nb, 1.0)
        n = xp.concatenate([na, nb], axis=1)
        vbar = xp.einsum("wn, in->wi", n, self.v)
        self.timer.tfbias += time.time() - start_time

        # Bernoulli Distributed auxillary fields
        start_time = time.time()
        Ta, Tb, cfb_log, cmf_log = self.construct_VHS(vbar)
        synchronize()
        self.timer.tvhs += time.time() - start_time

        # propagate
        start_time = time.time()

        propagator_a = xp.exp(Ta)
        propagator_b = xp.exp(Tb)

        apply_diag_oper(
            phia_transformed, propagator_a
        )  # apply diagonal propagator in extended space
        apply_diag_oper(phib_transformed, propagator_b)

        walkers.phia = apply_isometry(
            phia_transformed, self.isometry, reverse=False
        )  # rotate back to original space
        walkers.phib = apply_isometry(phib_transformed, self.isometry, reverse=False)

        synchronize()

        self.timer.tgemm += time.time() - start_time

        return cmf_log, cfb_log  # this is an array with dimension nwalkers

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift):
        synchronize()
        start_time = time.time()
        ovlp = trial.calc_overlap(walkers)  # greens function but only need overlap
        synchronize()
        self.timer.tovlp += time.time() - start_time  # ovlp function timer

        # 2. Update Slater matrix
        # 2.a Apply one-body
        self.propagate_walkers_one_body(walkers)

        # 2.b Apply two-body
        cmf, cfb = self.propagate_walkers_two_body(walkers, hamiltonian, trial)

        # 2.c Apply one-body
        self.propagate_walkers_one_body(walkers)

        # Now apply phaseless approximation
        start_time = time.time()
        ovlp_new = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_weight(walkers, ovlp, ovlp_new, cfb, cmf, eshift)
        synchronize()
        self.timer.tupdate += time.time() - start_time
