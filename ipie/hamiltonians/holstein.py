import numpy
from ipie.hamiltonians.generic_base import GenericBase
from ipie.utils.backend import arraylib as xp


class HolsteinModel(GenericBase):
    """Class for Holsteib model carrying elph tensor and system parameters
    """

    def __init__(self, g: float, t: float, w0: float, nsites: int, pbc: bool):
        self.g = g
        self.t = t
        self.w0 = w0
        self.m = 1/self.w0
        self.nsites = nsites
        self.pbc = pbc
        self.T = None
        self.const = -self.g * numpy.sqrt(2. * self.m * self.w0)

    def build(self):
        self.T = numpy.diag(numpy.ones(self.nsites-1), 1)
        self.T += numpy.diag(numpy.ones(self.nsites-1), -1)
    
        if self.pbc:
            self.T[0,-1] = self.T[-1,0] = 1.

        self.T *= -self.t

        self.T = [self.T.copy(), self.T.copy()]


