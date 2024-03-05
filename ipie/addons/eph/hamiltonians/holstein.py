import numpy
from ipie.utils.backend import arraylib as xp


class HolsteinModel():
    r"""Class carrying parameters specifying a 1D Holstein chain.

    The Holstein model is described by the Hamiltonian

    .. math::
        \hat{H} = -t \sum_{\langle ij\rangle} \hat{a}_i^\dagger \hat{a}_j 
        - g \sqrt{2 w_0 m} \sum_i \hat{a}_i^\dagger \hat{a}_i \hat{X}_i
        + \bigg(\sum_i \frac{m w_0^2}{2} \hat{X}_i^2 + \frac{1}{2m} \hat{P}_i^2 
        - \frac{w_0}{2}\bigg),

    where :math:`t` is associated with the electronic hopping, :math:`g` with
    the electron-phonon coupling strength, and :math:``w_0` with the phonon 
    frequency. 

    Parameters
    ----------
    g : 
        Electron-phonon coupling strength
    t : 
        Electron hopping parameter
    w0 : 
        Phonon frequency
    nsites : 
        Length of the 1D Holstein chain
    pbc : 
        Boolean specifying whether periodic boundary conditions should be 
        employed.
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
        """Constructs electronic hopping matrix."""
        self.T = numpy.diag(numpy.ones(self.nsites-1), 1)
        self.T += numpy.diag(numpy.ones(self.nsites-1), -1)
    
        if self.pbc:
            self.T[0,-1] = self.T[-1,0] = 1.

        self.T *= -self.t

        self.T = [self.T.copy(), self.T.copy()]


