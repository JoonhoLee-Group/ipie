import numpy
import scipy.linalg

from ipie.addons.thermal.estimators.particle_number import particle_number
from ipie.addons.thermal.estimators.thermal import one_rdm_stable
from ipie.addons.thermal.estimators.greens_function import greens_function
from ipie.addons.thermal.trial.chem_pot import compute_rho, find_chemical_potential
from ipie.utils.misc import update_stack


class OneBody(object):
    def __init__(self, hamiltonian, nelec, beta, dt, options=None,
                 alt_convention=False, H1=None, verbose=False):
        if options is None:
            options = {}

        self.name = "thermal"
        self.compute_trial_energy = False
        self.verbose = verbose
        self.alt_convention = alt_convention

        if H1 is None:
            try:
                self.H1 = hamiltonian.H1

            except AttributeError:
                self.H1 = hamiltonian.h1e
                
        else:
            self.H1 = H1

        if verbose:
            print("# Building OneBody density matrix.")
            print(f"# beta in OneBody: {beta}")
            print(f"# dt in OneBody: {dt}")

        dmat_up = scipy.linalg.expm(-dt * (self.H1[0]))
        dmat_down = scipy.linalg.expm(-dt * (self.H1[1]))
        self.dmat = numpy.array([dmat_up, dmat_down])
        cond = numpy.linalg.cond(self.dmat[0])

        if verbose:
            print(f"# condition number of BT: {cond: 10e}")
        
        self.nelec = nelec
        self.nav = options.get("nav", None)

        if self.nav is None:
            self.nav = numpy.sum(self.nelec)

        if verbose:
            print(f"# Target average electron number: {self.nav}")

        self.max_it = options.get("max_it", 1000)
        self.deps = options.get("threshold", 1e-6)
        self.mu = options.get("mu", None)

        self.nslice = int(beta / dt)
        self.stack_size = options.get("stack_size", None)

        if self.stack_size == None:
            if verbose:
                print("# Estimating stack size from BT.")

            self.cond = numpy.linalg.cond(self.dmat[0])
            # We will end up multiplying many BTs together. Can roughly determine
            # safe stack size from condition number of BT as the condition number of
            # the product will scale roughly as cond(BT)^(number of products).
            # We can determine a conservative stack size by requiring that the
            # condition number of the product does not exceed 1e3.
            self.stack_size = min(self.nslice, int(3.0 / numpy.log10(self.cond)))

            if verbose:
                print("# Initial stack size, # of slices: {}, {}".format(
                    self.stack_size, self.nslice))

        # Adjust stack size
        self.stack_size = update_stack(self.stack_size, self.nslice, verbose=verbose)
        self.stack_length = int(beta / (self.stack_size * dt))

        if verbose:
            print(f"# Number of stacks: {self.stack_length}")

        sign = 1
        if self.alt_convention:
            if verbose:
                print("# Using alternate sign convention for chemical potential.")

            sign = -1

        self.dtau = self.stack_size * dt

        if self.mu is None:
            self.rho = numpy.array([scipy.linalg.expm(-self.dtau * (self.H1[0])),
                                    scipy.linalg.expm(-self.dtau * (self.H1[1]))])
            self.mu = find_chemical_potential(
                        self.alt_convention, self.rho, self.dtau, self.stack_length,
                        self.nav, deps=self.deps, max_it=self.max_it, verbose=verbose)
            
        else:
            self.rho = numpy.array([scipy.linalg.expm(-self.dtau * (self.H1[0])),
                                    scipy.linalg.expm(-self.dtau * (self.H1[1]))])

        if self.verbose:
            print(f"# Chemical potential in trial density matrix: {self.mu: .10e}")

        self.P = one_rdm_stable(compute_rho(self.rho, self.mu, self.dtau, sign=sign), self.stack_length)
        self.nav = particle_number(self.P).real

        if self.verbose:
            print(f"# Average particle number in trial density matrix: {self.nav}")

        self.dmat = compute_rho(self.dmat, self.mu, dt, sign=sign)
        self.dmat_inv = numpy.array([scipy.linalg.inv(self.dmat[0], check_finite=False),
                                     scipy.linalg.inv(self.dmat[1], check_finite=False)])

        self.G = numpy.array([greens_function(self.dmat[0]), greens_function(self.dmat[1])])
        self.error = False
        self.init = numpy.array([0])
