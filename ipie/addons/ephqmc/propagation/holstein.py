import numpy 
import time
import scipy.linalg

from ipie.addons.ephqmc.hamiltonians.holstein import HolsteinModel
from ipie.propagation.operations import propagate_one_body
from ipie.utils.backend import synchronize, cast_to_device
from ipie.propagation.continuous_base import PropagatorTimer

def construct_one_body_propagator(hamiltonian: HolsteinModel, dt: float):
    """"""
    H1 = hamiltonian.T
    expH1 = numpy.array(
        [scipy.linalg.expm(-0.5 * dt * H1[0]), scipy.linalg.expm(-0.5 * dt * H1[1])]
    )
    return expH1


class HolsteinPropagatorFree():
    """"""
    def __init__(self, time_step, verbose=False):
        self.dt = time_step
        self.verbose = verbose
        self.timer = PropagatorTimer()

        self.sqrt_dt = self.dt**0.5
        self.dt_ph = 0.5 * self.dt
        self.mpi_handler = None

    def build(self, hamiltonian) :   
        self.expH1 = construct_one_body_propagator(hamiltonian, self.dt)
        self.const = hamiltonian.g * numpy.sqrt(2. * hamiltonian.m * hamiltonian.w0) * self.dt
        self.w0 = hamiltonian.w0
        self.m = hamiltonian.m
        self.scale = numpy.sqrt(self.dt_ph / self.m)
        self.nsites = hamiltonian.nsites

    def propagate_walkers_one_body(self, walkers):
        start_time = time.time()
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        if walkers.ndown > 0 and not walkers.rhf:
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
        synchronize()
        self.timer.tgemm += time.time() - start_time

    def propagate_phonons(self, walkers):
        start_time = time.time()

        pot = 0.25 * self.m * self.w0**2 * numpy.sum(walkers.x**2, axis=1)
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot)

        N = numpy.random.normal(loc=0.0, scale=self.scale, 
                             size=(walkers.nwalkers, self.nsites))        
        walkers.x = walkers.x + N 

        pot = 0.25 * self.m * self.w0**2 * numpy.sum(walkers.x**2, axis=1)
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot)
            
        walkers.weight *= numpy.exp(self.dt_ph * self.nsites * self.w0 / 2) #doesnt matter for estimators

        synchronize()
        self.timer.tgemm += time.time() - start_time

    def propagate_electron(self, walkers, trial):
        start_time = time.time()
        ovlp = trial.calc_greens_function(walkers) 
        synchronize()
        self.timer.tgf += time.time() - start_time

        expEph = numpy.exp(self.const * walkers.x)
        
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        walkers.phia = numpy.einsum('ni,nie->nie', expEph, walkers.phia)
        walkers.phia = propagate_one_body(walkers.phia, self.expH1[0])
        
        if walkers.ndown > 0:
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])
            walkers.phib = numpy.einsum('ni,nie->nie', expEph, walkers.phib)
            walkers.phib = propagate_one_body(walkers.phib, self.expH1[1])

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift=None):
        synchronize()
        start_time = time.time()
        ovlp = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tgf += time.time() - start_time

        # 2. Update Walkers
        # 2.a DMC for phonon degrees of freedom
        self.propagate_phonons(walkers)

        # 2.b One-body propagation for electrons
        self.propagate_electron(walkers, trial)

        # 2.c DMC for phonon degrees of freedom
        self.propagate_phonons(walkers)

        # Update weights (and later do phaseless for multi-electron)
        start_time = time.time()
        ovlp_new = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        self.update_weight(walkers, ovlp, ovlp_new)
        synchronize()
        self.timer.tupdate += time.time() - start_time

    
    def update_weight(self, walkers, ovlp, ovlp_new):
        walkers.weight *= ovlp_new / ovlp


class HolsteinPropagatorImportance(HolsteinPropagatorFree):
    """"""
    def __init__(self, time_step, verbose=False):
        super().__init__(time_step, verbose=verbose)

    def propagate_phonons(self, walkers, hamiltonian, trial):
        """Propagates phonons via DMC"""
        start_time = time.time()
        #no ZPE in pot -> cancels with ZPE of etrial, wouldn't affect estimators anyways
        ph_ovlp_old = trial.calc_phonon_overlap(walkers)
        
        pot = 0.5 * hamiltonian.m * hamiltonian.w0**2 * numpy.sum(walkers.x**2, axis=1)
        pot -= 0.5 * trial.calc_phonon_laplacian_importance(walkers) / hamiltonian.m
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot / 2)

        N = numpy.random.normal(loc=0.0, scale=self.scale, size=(walkers.nwalkers, self.nsites))    
        drift = trial.calc_phonon_gradient(walkers)
        walkers.x = walkers.x + N + self.dt_ph * drift / hamiltonian.m

        ph_ovlp_new = trial.calc_phonon_overlap(walkers)        

        pot = 0.5 * hamiltonian.m * hamiltonian.w0**2 * numpy.sum(walkers.x**2, axis=1)
        pot -= 0.5 * trial.calc_phonon_laplacian_importance(walkers) / hamiltonian.m
        pot = numpy.real(pot)
        walkers.weight *= numpy.exp(-self.dt_ph * pot / 2)

        walkers.weight *= ph_ovlp_old / ph_ovlp_new
        walkers.weight *= numpy.exp(self.dt_ph * trial.energy)

        synchronize()
        self.timer.tgemm += time.time() - start_time
        

    def propagate_walkers(self, walkers, hamiltonian, trial, eshift=None):
        """"""
        synchronize()
        start_time = time.time()
        
        ovlp = trial.calc_overlap(walkers).copy()
        
        synchronize()
        self.timer.tgf += time.time() - start_time

        # 2. Update Walkers
        # 2.a DMC for phonon degrees of freedom
        self.propagate_phonons(walkers, hamiltonian, trial)
        
        # 2.b One-body propagation for electrons
        self.propagate_electron(walkers, trial)
        
        # 2.c DMC for phonon degrees of freedom
        self.propagate_phonons(walkers, hamiltonian, trial)

        start_time = time.time()
        
        ovlp_new = trial.calc_overlap(walkers)
        synchronize()
        self.timer.tovlp += time.time() - start_time

        start_time = time.time()
        
        self.update_weight(walkers, ovlp, ovlp_new)
        
        synchronize()
        self.timer.tupdate += time.time() - start_time

