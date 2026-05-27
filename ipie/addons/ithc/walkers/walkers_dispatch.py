import numpy

from ipie.addons.ithc.trial_wavefunction.single_det import SingleDet
from ipie.utils.mpi import MPIHandler
from ipie.walkers.ghf_walkers import GHFWalkers
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.walkers_dispatch import (
    get_initial_walker as _core_get_initial_walker,
    UHFWalkersTrial as _core_uhf_walkers_trial,
)


def get_initial_walker(trial) -> numpy.ndarray:
    if isinstance(trial, SingleDet):
        return 1, trial.psi.copy()
    return _core_get_initial_walker(trial)


def GHFWalkersTrial(
    trial,
    initial_walker: numpy.ndarray,
    nup: int,
    ndown: int,
    nbasis: int,
    nwalkers: int,
    mpi_handler: MPIHandler,
    verbose: bool = False,
):
    return GHFWalkers(initial_walker, nup, ndown, nbasis, nwalkers, mpi_handler, verbose=verbose)


def UHFWalkersTrial(
    trial,
    initial_walker: numpy.ndarray,
    nup: int,
    ndown: int,
    nbasis: int,
    *args,
    verbose: bool = False,
):
    if isinstance(trial, SingleDet):
        if len(args) not in (2, 3):
            raise TypeError("UHF walkers require nwalkers and mpi_handler")
        nwalkers, mpi_handler = args[:2]
        if len(args) == 3:
            verbose = args[2]
        return UHFWalkers(
            initial_walker,
            nup,
            ndown,
            nbasis,
            nwalkers,
            mpi_handler,
            verbose=verbose,
        )

    if len(args) == 3:
        return _core_uhf_walkers_trial(trial, initial_walker, nup, ndown, nbasis, *args)
    return _core_uhf_walkers_trial(
        trial, initial_walker, nup, ndown, nbasis, *args, verbose=verbose
    )
