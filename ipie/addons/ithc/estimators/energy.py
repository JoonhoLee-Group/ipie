import plum

from ipie.addons.ithc.estimators.local_energy_ithc import local_energy_single_det_uhf_ithc
from ipie.addons.ithc.hamiltonians.generic_ithc import GenericITHC
from ipie.addons.ithc.trial_wavefunction.single_det import SingleDet
from ipie.estimators.energy import EnergyEstimator, local_energy
from ipie.systems.generic import Generic
from ipie.walkers.uhf_walkers import UHFWalkers


@plum.dispatch
def local_energy(
    system: Generic,
    hamiltonian: GenericITHC,
    walkers: UHFWalkers,
    trial: SingleDet,
):
    return local_energy_single_det_uhf_ithc(system, hamiltonian, walkers, trial)
