import math
import sys

import numpy
import scipy.linalg

from ipie.legacy.estimators.local_energy import local_energy
from ipie.legacy.estimators.thermal import (greens_function, one_rdm,
                                            one_rdm_from_G, one_rdm_stable,
                                            particle_number)
from ipie.legacy.trial_density_matrices.chem_pot import (
    compute_rho, find_chemical_potential)
from ipie.utils.misc import update_stack


class OneBody(object):
    def __init__(
        self,
        system,
        hamiltonian,
        beta,
        dt,
        options={},
        nav=None,
        H1=None,
        verbose=False,
    ):
        self.name = "thermal"
        self.verbose = verbose
        if H1 is None:
            try:
                self.H1 = hamiltonian.H1
            except AttributeError:
                self.H1 = hamiltonian.h1e
        else:
            self.H1 = H1

        if verbose:
            print("# Building OneBody density matrix.")
            print("# beta in OneBody: {}".format(beta))
            print("# dt in OneBody: {}".format(dt))

        dmat_up = scipy.linalg.expm(-dt * (self.H1[0]))
        dmat_down = scipy.linalg.expm(-dt * (self.H1[1]))
        self.dmat = numpy.array([dmat_up, dmat_down])
        cond = numpy.linalg.cond(self.dmat[0])
        if verbose:
            print("# condition number of BT: {: 10e}".format(cond))

        if nav is not None:
            self.nav = nav
        else:
            self.nav = options.get("nav", None)
            if self.nav is None:
                self.nav = system.nup + system.ndown
        if verbose:
            print("# Target average electron number: {}".format(self.nav))

        self.max_it = options.get("max_it", 1000)
        self.deps = options.get("threshold", 1e-6)
        self.mu = options.get("mu", None)

        self.num_slices = int(beta / dt)
        self.stack_size = options.get("stack_size", None)

        if self.stack_size == None:
            if verbose:
                print("# Estimating stack size from BT.")
            eigs, ev = scipy.linalg.eigh(self.dmat[0])
            emax = numpy.max(eigs)
            emin = numpy.min(eigs)
            self.cond = numpy.linalg.cond(self.dmat[0])
            # We will end up multiplying many BTs together. Can roughly determine
            # safe stack size from condition number of BT as the condition number of
            # the product will scale roughly as cond(BT)^(number of products).
            # We can determine a conservative stack size by requiring that the
            # condition number of the product does not exceed 1e3.
            self.stack_size = min(self.num_slices, int(3.0 / numpy.log10(self.cond)))
            if verbose:
                print(
                    "# Initial stack size, # of slices: {}, {}".format(
                        self.stack_size, self.num_slices
                    )
                )

        # adjust stack size
        self.stack_size = update_stack(
            self.stack_size, self.num_slices, verbose=verbose
        )
        self.num_bins = int(beta / (self.stack_size * dt))

        if verbose:
            print("# Number of stacks: {}".format(self.num_bins))

        sign = 1
        if hamiltonian._alt_convention:
            if verbose:
                print("# Using alternate sign convention for chemical potential.")
            sign = -1
        dtau = self.stack_size * dt
        self.dtau = dtau

        if self.mu is None:
            self.rho = numpy.array(
                [
                    scipy.linalg.expm(-dtau * (self.H1[0])),
                    scipy.linalg.expm(-dtau * (self.H1[1])),
                ]
            )
            self.mu = find_chemical_potential(
                hamiltonian._alt_convention,
                self.rho,
                dtau,
                self.num_bins,
                self.nav,
                deps=self.deps,
                max_it=self.max_it,
                verbose=verbose,
            )
        else:
            self.rho = numpy.array(
                [
                    scipy.linalg.expm(-dtau * (self.H1[0])),
                    scipy.linalg.expm(-dtau * (self.H1[1])),
                ]
            )

        if self.verbose:
            print(
                "# Chemical potential in trial density matrix: {: .10e}".format(self.mu)
            )

        self.P = one_rdm_stable(
            compute_rho(self.rho, self.mu, dtau, sign=sign), self.num_bins
        )
        self.nav = particle_number(self.P).real
        if self.verbose:
            print(
                "# Average particle number in trial density matrix: "
                "{}".format(self.nav)
            )
        self.dmat = compute_rho(self.dmat, self.mu, dt, sign=sign)
        self.dmat_inv = numpy.array(
            [
                scipy.linalg.inv(self.dmat[0], check_finite=False),
                scipy.linalg.inv(self.dmat[1], check_finite=False),
            ]
        )

        self.G = numpy.array(
            [greens_function(self.dmat[0]), greens_function(self.dmat[1])]
        )
        self.error = False
        self.init = numpy.array([0])
