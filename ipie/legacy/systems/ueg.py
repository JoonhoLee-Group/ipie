import math
import sys
import time

import numpy
import scipy.linalg
import scipy.sparse

import ipie.utils
from ipie.utils.io import write_qmcpack_sparse


class UEG(object):
    """UEG system class (integrals read from fcidump)
    Parameters
    ----------
    nup : int
        Number of up electrons.
    ndown : int
        Number of down electrons.
    rs : float
        Density parameter.
    ecut : float
        Scaled cutoff energy.
    ktwist : :class:`numpy.ndarray`
        Twist vector.
    verbose : bool
        Print extra information.
    Attributes
    ----------
    T : :class:`numpy.ndarray`
        One-body part of the Hamiltonian. This is diagonal in plane wave basis.
    ecore : float
        Madelung contribution to the total energy.
    h1e_mod : :class:`numpy.ndarray`
        Modified one-body Hamiltonian.
    nfields : int
        Number of field configurations per walker for back propagation.
    basis : :class:`numpy.ndarray`
        Basis vectors within a cutoff.
    kfac : float
        Scale factor (2pi/L).
    """

    def __init__(self, options, verbose=False):
        if verbose:
            print("# Parsing input options for systems.UEG.")
        self.name = "UEG"
        self.nup = options.get("nup")
        self.ndown = options.get("ndown")
        self.nelec = (self.nup, self.ndown)
        self.rs = options.get("rs")
        self.ecut = options.get("ecut")
        self.ktwist = numpy.array(options.get("ktwist", [0, 0, 0])).reshape(3)
        self.control_variate = False
        self.mu = options.get("mu", None)
        if verbose:
            print("# Number of spin-up electrons: {:d}".format(self.nup))
            print("# Number of spin-down electrons: {:d}".format(self.ndown))
            print("# rs: {:6.4e}".format(self.rs))
            if self.mu is not None:
                print("# mu: {:6.4e}".format(self.mu))

        self.thermal = options.get("thermal", False)
        if self.thermal and verbose:
            print("# Thermal UEG activated")
        self._alt_convention = options.get("alt_convention", False)
        self.sparse = True

        # total # of electrons
        self.ne = self.nup + self.ndown
        # spin polarisation
        self.zeta = (self.nup - self.ndown) / self.ne
        # Density.
        self.rho = ((4.0 * math.pi) / 3.0 * self.rs**3.0) ** (-1.0)
        # Box Length.
        self.L = self.rs * (4.0 * self.ne * math.pi / 3.0) ** (1 / 3.0)
        # Volume
        self.vol = self.L**3.0
        # k-space grid spacing.
        # self.kfac = 2*math.pi/self.L
        self.kfac = 2 * math.pi / self.L
        # Fermi Wavevector (infinite system).
        self.kf = (3 * (self.zeta + 1) * math.pi**2 * self.ne / self.L**3) ** (
            1 / 3.0
        )
        # Fermi energy (inifinite systems).
        self.ef = 0.5 * self.kf**2
        if verbose:
            print("# Spin polarisation (zeta): {:6.4e}".format(self.zeta))
            print("# Electron density (rho): {:13.8e}".format(self.rho))
            print("# Box Length (L): {:13.8e}".format(self.L))
            print("# Volume: {:13.8e}".format(self.vol))
            print("# k-space factor (2pi/L): {:13.8e}".format(self.kfac))


def unit_test():
    from numpy import linalg as LA

    from ipie.estimators import ci as pieci

    options = {"nup": 2, "ndown": 2, "rs": 1.0, "thermal": True, "ecut": 3}
    system = UEG(options, True)


if __name__ == "__main__":
    unit_test()
