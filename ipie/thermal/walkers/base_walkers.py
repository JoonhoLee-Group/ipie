import numpy
from ipie.walkers.base_walkers import WalkerAccumulator


class ThermalWalkerAccumulator(WalkerAccumulator):
    def __init__(self, names, nsteps):
        super().__init__(names, nsteps)

    def zero(self, walkers, trial):
        self.buffer.fill(0.j)
        walkers.weight = numpy.ones(walkers.nwalkers)
        walkers.phase = numpy.ones(walkers.nwalkers, dtype=numpy.complex128)
        
        for iw in range(walkers.nwalkers):
            walkers.stack[iw].reset()
            walkers.stack[iw].set_all(trial.dmat)
            walkers.calc_greens_function(iw)
