import numpy 
import time
import scipy.linalg

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.propagation.operations import propagate_one_body
from ipie.utils.backend import synchronize, cast_to_device
from ipie.propagation.continuous_base import PropagatorTimer

def construct_one_body_propagator(hamiltonian: HolsteinModel, dt: float):
    """Exponentiates the electronic hopping term to apply it later as
    part of the trotterized algorithm.

    Parameters
    ----------
    hamiltonian :
        Hamiltonian caryying the one-body term as hamiltonian.T
    dt : 
        Time step
    """
    H1 = hamiltonian.T
    expH1 = numpy.array(
        [scipy.linalg.expm(-0.5 * dt * H1[0]), scipy.linalg.expm(-0.5 * dt * H1[1])]
    )
    return expH1


class HolsteinPropagatorFree:
    r"""Propagates walkers by trotterization,
    .. math:: 
        \mathrm{e}^{-\Delta \tau \hat{H}} \approx \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{ph}} / 2} 
        \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el}} / 2} \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el-ph}}} 
        \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el}} / 2} \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{ph}} / 2},

    where propagation under :math:`\hat{H}_{\mathrm{ph}}` employs a generic 
    Diffucion MC procedure (notably without importance sampling). Propagation by 
    :math:`\hat{H}_{\mathrm{el}}` consists of a simple mat-vec. As 
    :math:`\hat{H}_{\mathrm{el-ph}}` is diagonal in bosonic position space we 
    can straightforwardly exponentiate the displacements and perform another
    mat-vec with this diagonal matrix apllied to electronic degrees of freedom.

    Parameters
    ----------
    time_step : 
        Time step
    verbose : 
        Print level
    """
    def __init__(self, time_step: float, verbose: bool = False):
        self.dt = time_step
        self.verbose = verbose
        self.timer = PropagatorTimer()

        self.sqrt_dt = self.dt**0.5
        self.dt_ph = 0.5 * self.dt
        self.mpi_handler = None

    def build(self, hamiltonian: HolsteinModel, trial=None, 
              walkers=None, mpi_handler=None) -> None:   
        """Necessary step before running the AFQMC procedure. 
        Sets required attributes. 
        
        Parameters
        ----------
        hamiltonian : 
            Holstein model
        trial :
            Trial class
        walkers : 
            Walkers class
        mpi_handler :
            MPIHandler specifying rank and size
        """
        self.expH1 = construct_one_body_propagator(hamiltonian, self.dt)
        self.const = hamiltonian.g * numpy.sqrt(2. * hamiltonian.m * hamiltonian.w0) * self.dt
        self.w0 = hamiltonian.w0
        self.m = hamiltonian.m
        self.scale = numpy.sqrt(self.dt_ph / self.m)
        self.nsites = hamiltonian.nsites

    def propagate_phonons(self, walkers, hamiltonian, trial) -> None:
        r"""Propagates phonon displacements by adjusting weigths according to
        bosonic on-site energies and sampling the momentum contribution, again
        by trotterizing the phonon propagator.
        
        .. math:: 
            \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{ph}} / 2} \approx 
            \mathrm{e}^{\Delta \tau N \omega / 4} 
            \mathrm{e}^{-\Delta \tau \sum_i m \omega \hat{X}_i^2 / 8}
            \mathrm{e}^{-\Delta \tau \sum_i \hat{P}_i^2 / (4 \omega)} 
            \mathrm{e}^{-\Delta \tau \sum_i m \omega \hat{X}_i^2 / 8} 

        One can obtain the sampling prescription by insertion of resolutions of
        identity, :math:`\int dX |X\rangle \langleX|, and performin the resulting
        Fourier transformation. 

        Parameters
        ----------
        walkers : 
            Walkers class
        """
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
            
        # Does not matter for estimators but helps with population control
        walkers.weight *= numpy.exp(self.dt_ph * self.nsites * self.w0 / 2)

        synchronize()
        self.timer.tgemm += time.time() - start_time

    def propagate_electron(self, walkers, hamiltonian, trial) -> None:
        r"""Propagates electronic degrees of freedom via

        .. math:: 
            \mathrm{e}^{-\Delta \tau (\hat{H}_{\mathrm{el}} \otimes \hat{I}_{\mathrm{ph}} + \hat{H}_{\mathrm{el-ph}})} 
            \approx \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el}} / 2}
            \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el-ph}}}
            \mathrm{e}^{-\Delta \tau \hat{H}_{\mathrm{el}} / 2}.

        This acts on walkers of the form :math:`|\phi(\tau)\rangle \otimes |X(\tau)\rangle`.

        
        Parameters
        ----------
        walkers : 
            Walkers class
        trial : 
            Trial class
        """
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

        # Update Walkers
        # a) DMC for phonon degrees of freedom
        self.propagate_phonons(walkers, hamiltonian, trial)

        # b) One-body propagation for electrons
        self.propagate_electron(walkers, hamiltonian, trial)

        # c) DMC for phonon degrees of freedom
        self.propagate_phonons(walkers, hamiltonian, trial)

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


class HolsteinPropagator(HolsteinPropagatorFree):
    r"""Propagates walkers by trotterization, employing importance sampling for 
    the bosonic degrees of freedom. This results in a different weigth update,
    and the additional displacement update by the drift term,
    
    .. math::
        D = \frac{\nabla_X \langle \Psi_\mathrm{T} | \psi(\tau), X(\tau)\rangle}
        {\langle \Psi_\mathrm{T} | \psi(\tau), X(\tau)\rangle},

    such that the revised displacement update reads

    .. math:: 
        X(\tau+\Delta\tau) = X(\tau) 
        + \mathcal{N}(\mu=0, \sigma = \sqrt{\frac{\Delta\tau}{m}}) 
        + \frac{\Delta\tau}{m} D.

    Parameters
    ----------
    time_step : 
        Time step
    verbose :
        Print level
    """
    def __init__(self, time_step, verbose=False):
        super().__init__(time_step, verbose=verbose)

    def propagate_phonons(self, walkers, hamiltonian, trial):
        """Propagates phonons via Diffusion MC including drift term."""
        start_time = time.time()
        
        # No ZPE in pot -> cancels with ZPE of etrial, wouldn't affect estimators anyways
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
        

