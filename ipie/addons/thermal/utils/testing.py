import numpy
import pytest
from typing import Tuple, Union

from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric

from ipie.addons.thermal.utils.ueg import UEG
from ipie.addons.thermal.trial.one_body import OneBody
from ipie.addons.thermal.trial.mean_field import MeanField
from ipie.addons.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.addons.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.addons.thermal.qmc.options import ThermalQMCParams
from ipie.addons.thermal.qmc.thermal_afqmc import ThermalAFQMC


def build_generic_test_case_handlers(
        nelec: Tuple[int, int],
        nbasis: int,
        mu: float,
        beta: float,
        timestep: float,
        nwalkers: int = 100,
        stack_size: int = 10,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        diagonal: bool = False,
        mf_trial: bool = True,
        propagate: bool = False,
        complex_integrals: bool = False,
        debug: bool = False,
        with_eri: bool = False,
        seed: Union[int, None] = None,
        verbose: int = 0):
    sym = 8
    if complex_integrals: sym = 4
    numpy.random.seed(seed)
    
    # 1. Generate random integrals.
    h1e, chol, _, eri = generate_hamiltonian(nbasis, nelec, cplx=complex_integrals, 
                                             sym=sym, tol=1e-10)

    if diagonal:
        h1e = numpy.diag(numpy.diag(h1e))
    
    # 2. Build Hamiltonian.
    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]),
                             chol=chol.reshape((-1, nbasis**2)).T.copy(),
                             ecore=0)
    
    # 3. Build trial.
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    if mf_trial:
        trial = MeanField(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # 4. Build walkers.
    walkers = UHFThermalWalkers(
                trial, nbasis, nwalkers, stack_size=stack_size, lowrank=lowrank, 
                lowrank_thresh=lowrank_thresh, verbose=verbose)
    
    # 5. Build propagator.
    propagator = PhaselessGeneric(timestep, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)

    if propagate:
        for t in range(walkers.stack[0].nslice):
            propagator.propagate_walkers(walkers, hamiltonian, trial, debug=debug)

    objs = {'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers,
            'propagator': propagator}

    if with_eri:
        objs['eri'] = eri

    return objs


def build_generic_test_case_handlers_mpi(
        nelec: Tuple[int, int],
        nbasis: int,
        mu: float,
        beta: float,
        timestep: float,
        mpi_handler: MPIHandler,
        nwalkers: int = 100,
        stack_size: int = 10,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        diagonal: bool = False,
        mf_trial: bool = True,
        propagate: bool = False,
        complex_integrals: bool = False,
        debug: bool = False,
        seed: Union[int, None] = None,
        verbose: int = 0):
    sym = 8
    if complex_integrals: sym = 4
    numpy.random.seed(seed)
    
    # 1. Generate random integrals.
    h1e, chol, _, _ = generate_hamiltonian(nbasis, nelec, cplx=complex_integrals, 
                                           sym=sym, tol=1e-10)

    if diagonal:
        h1e = numpy.diag(numpy.diag(h1e))
    
    # 2. Build Hamiltonian.
    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]),
                             chol=chol.reshape((-1, nbasis**2)).T.copy(),
                             ecore=0)
    
    # 3. Build trial.
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    if mf_trial:
        trial = MeanField(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # 4. Build walkers.
    walkers = UHFThermalWalkers(
                trial, nbasis, nwalkers, stack_size=stack_size, lowrank=lowrank, 
                lowrank_thresh=lowrank_thresh, mpi_handler=mpi_handler, verbose=verbose)
    
    # 5. Build propagator.
    propagator = PhaselessGeneric(timestep, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, 
                     mpi_handler=mpi_handler, verbose=verbose)

    if propagate:
        for t in range(walkers.stack[0].nslice):
            propagator.propagate_walkers(walkers, hamiltonian, trial, debug=debug)

    objs = {'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers,
            'propagator': propagator}
    return objs


def build_driver_generic_test_instance(
        nelec: Tuple[int, int],
        nbasis: int,
        mu: float,
        beta: float,
        timestep: float,
        nblocks: int,
        nwalkers: int = 100,
        stack_size: int = 10,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        stabilize_freq: int = 5,
        pop_control_freq: int = 5,
        pop_control_method: str = 'pair_branch',
        diagonal: bool = False,
        complex_integrals: bool = False,
        debug: bool = False,
        seed: Union[int, None] = None,
        verbose: int = 0):
    sym = 8
    if complex_integrals: sym = 4
    numpy.random.seed(seed)

    # 1. Generate random integrals.
    h1e, chol, _, _ = generate_hamiltonian(nbasis, nelec, cplx=complex_integrals, 
                                           sym=sym, tol=1e-10)
    
    if diagonal:
        h1e = numpy.diag(numpy.diag(h1e))
    
    # 2. Build Hamiltonian.
    hamiltonian = HamGeneric(h1e=numpy.array([h1e, h1e]),
                             chol=chol.reshape((-1, nbasis**2)).T.copy(),
                             ecore=0)
    
    # 3. Build trial.
    trial = MeanField(hamiltonian, nelec, beta, timestep)

    # 4. Build Thermal AFQMC driver.
    afqmc = ThermalAFQMC.build(
                nelec, mu, beta, hamiltonian, trial, nwalkers=nwalkers, 
                stack_size=stack_size, seed=seed, nblocks=nblocks, timestep=timestep, 
                stabilize_freq=stabilize_freq, pop_control_freq=pop_control_freq,
                pop_control_method=pop_control_method, lowrank=lowrank, 
                lowrank_thresh=lowrank_thresh, debug=debug, verbose=verbose)
    return afqmc


def build_ueg_test_case_handlers(
        nelec: Tuple[int, int],
        rs: float,
        ecut: float,
        mu: float,
        beta: float,
        timestep: float,
        nwalkers: int = 100,
        stack_size: int = 10,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        propagate: bool = False,
        debug: bool = False,
        seed: Union[int, None] = None,
        verbose: int = 0):
    nup, ndown = nelec
    ueg_opts = {
                "nup": nup,
                "ndown": ndown,
                "rs": rs,
                "ecut": ecut,
                "thermal": True,
                "write_integrals": False,
                "low_rank": lowrank
                }

    numpy.random.seed(seed)

    # 1. Generate UEG integrals.
    ueg = UEG(ueg_opts, verbose=verbose)
    ueg.build(verbose=verbose)
    nbasis = ueg.nbasis
    nchol = ueg.nchol

    if verbose:
        print(f"# nbasis = {nbasis}")
        print(f"# nchol = {nchol}")
        print(f"# nup = {nup}")
        print(f"# ndown = {ndown}")
        
    h1 = ueg.H1[0]
    chol = 2. * ueg.chol_vecs.toarray().copy()
    ecore = 0.

    # 2. Build Hamiltonian.
    hamiltonian = HamGeneric(
            numpy.array([h1, h1], dtype=numpy.complex128), 
            numpy.array(chol, dtype=numpy.complex128), 
            ecore,
            verbose=verbose)

    # 3. Build trial.
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # 4. Build walkers.
    walkers = UHFThermalWalkers(
                trial, nbasis, nwalkers, stack_size=stack_size, lowrank=lowrank, 
                lowrank_thresh=lowrank_thresh, verbose=verbose)
    
    # 5. Build propagator.
    propagator = PhaselessGeneric(timestep, mu, lowrank=lowrank, verbose=verbose)
    propagator.build(hamiltonian, trial=trial, walkers=walkers, verbose=verbose)

    if propagate:
        for t in range(walkers.stack[0].nslice):
            propagator.propagate_walkers(walkers, hamiltonian, trial, debug=debug)

    objs = {'trial': trial,
            'hamiltonian': hamiltonian,
            'walkers': walkers,
            'propagator': propagator}
    return objs


def build_driver_ueg_test_instance(
        nelec: Tuple[int, int],
        rs: float,
        ecut: float,
        mu: float,
        beta: float,
        timestep: float,
        nblocks: int,
        nwalkers: int = 100,
        stack_size: int = 10,
        lowrank: bool = False,
        lowrank_thresh: float = 1e-6,
        stabilize_freq: int = 5,
        pop_control_freq: int = 5,
        pop_control_method: str = 'pair_branch',
        debug: bool = False,
        seed: Union[int, None] = None,
        verbose: int = 0):
    nup, ndown = nelec
    ueg_opts = {
                "nup": nup,
                "ndown": ndown,
                "rs": rs,
                "ecut": ecut,
                "thermal": True,
                "write_integrals": False,
                "low_rank": lowrank
                }

    numpy.random.seed(seed)

    # 1. Generate UEG integrals.
    ueg = UEG(ueg_opts, verbose=verbose)
    ueg.build(verbose=verbose)
    nbasis = ueg.nbasis
    nchol = ueg.nchol

    if verbose:
        print(f"# nbasis = {nbasis}")
        print(f"# nchol = {nchol}")
        print(f"# nup = {nup}")
        print(f"# ndown = {ndown}")
        
    h1 = ueg.H1[0]
    chol = 2. * ueg.chol_vecs.toarray().copy()
    ecore = 0.

    # 2. Build Hamiltonian.
    hamiltonian = HamGeneric(
            numpy.array([h1, h1], dtype=numpy.complex128), 
            numpy.array(chol, dtype=numpy.complex128), 
            ecore,
            verbose=verbose)

    # 3. Build trial.
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # 4. Build Thermal AFQMC driver.
    afqmc = ThermalAFQMC.build(
                nelec, mu, beta, hamiltonian, trial, nwalkers=nwalkers,
                stack_size=stack_size, seed=seed, nblocks=nblocks, timestep=timestep, 
                stabilize_freq=stabilize_freq, pop_control_freq=pop_control_freq,
                pop_control_method=pop_control_method, lowrank=lowrank, 
                lowrank_thresh=lowrank_thresh, debug=debug, verbose=verbose)
    return afqmc

