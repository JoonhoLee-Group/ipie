import numpy
import pytest
from typing import Union

from ipie.qmc.options import QMCOpts
from ipie.systems.generic import Generic
from ipie.utils.mpi import MPIHandler
from ipie.utils.testing import generate_hamiltonian
from ipie.hamiltonians.generic import Generic as HamGeneric

from ipie.thermal.utils.ueg import UEG
from ipie.thermal.trial.one_body import OneBody
from ipie.thermal.trial.mean_field import MeanField
from ipie.thermal.walkers.uhf_walkers import UHFThermalWalkers
from ipie.thermal.propagation.phaseless_generic import PhaselessGeneric
from ipie.thermal.qmc.options import ThermalQMCParams
from ipie.thermal.qmc.thermal_afqmc import ThermalAFQMC


def build_generic_test_case_handlers(options: dict,
                                     seed: Union[int, None],
                                     debug: bool = False,
                                     with_eri: bool = False,
                                     verbose: bool = False):
    # Unpack options
    nelec = options['nelec']
    nbasis = options['nbasis']
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    nsteps_per_block = options['nsteps_per_block']
    nblocks = options['nblocks']
    stabilize_freq = options['stabilize_freq']
    pop_control_freq = options['pop_control_freq']
    pop_control_method = options['pop_control_method']
    lowrank = options['lowrank']

    complex_integrals = options.get('complex_integrals', True)
    mf_trial = options.get('mf_trial', True)
    propagate = options.get('propagate', False)
    diagonal = options.get('diagonal', False)

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
    hamiltonian.name = options["hamiltonian"]["name"]
    hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    hamiltonian.sparse = options["hamiltonian"]["sparse"]
    
    # 3. Build trial.
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    if mf_trial:
        trial = MeanField(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # 4. Build walkers.
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank, 
                                verbose=verbose)
    
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


def build_generic_test_case_handlers_mpi(options: dict,
                                         mpi_handler: MPIHandler,
                                         seed: Union[int, None],
                                         debug: bool = False,
                                         verbose: bool = False):
    # Unpack options
    nelec = options['nelec']
    nbasis = options['nbasis']
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    nsteps_per_block = options['nsteps_per_block']
    nblocks = options['nblocks']
    stabilize_freq = options['stabilize_freq']
    pop_control_freq = options['pop_control_freq']
    pop_control_method = options['pop_control_method']
    lowrank = options['lowrank']

    complex_integrals = options.get('complex_integrals', True)
    mf_trial = options.get('mf_trial', True)
    propagate = options.get('propagate', False)
    diagonal = options.get('diagonal', False)

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
    hamiltonian.name = options["hamiltonian"]["name"]
    hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    hamiltonian.sparse = options["hamiltonian"]["sparse"]
    
    # 3. Build trial.
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    if mf_trial:
        trial = MeanField(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # 4. Build walkers.
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank, 
                                mpi_handler=mpi_handler, verbose=verbose)
    
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


def build_driver_generic_test_instance(options: Union[dict, None],
                                       seed: Union[int, None],
                                       debug: bool = False,
                                       verbose: bool = False):
    # Unpack options
    nelec = options['nelec']
    nbasis = options['nbasis']
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    nsteps_per_block = options['nsteps_per_block']
    nblocks = options['nblocks']
    stabilize_freq = options['stabilize_freq']
    pop_control_freq = options['pop_control_freq']
    pop_control_method = options['pop_control_method']
    lowrank = options['lowrank']
    complex_integrals = options['complex_integrals']
    diagonal = options.get('diagonal', False)

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
    hamiltonian.name = options["hamiltonian"]["name"]
    hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    hamiltonian.sparse = options["hamiltonian"]["sparse"]
    
    # 3. Build trial.
    trial = MeanField(hamiltonian, nelec, beta, timestep)

    # 4. Build Thermal AFQMC driver.
    afqmc = ThermalAFQMC.build(
                nelec, mu, beta, hamiltonian, trial, nwalkers, seed, 
                nblocks=nblocks, timestep=timestep, stabilize_freq=stabilize_freq,
                pop_control_freq=pop_control_freq, pop_control_method=pop_control_method,
                lowrank=lowrank, debug=debug, verbose=verbose)
    return afqmc


def build_ueg_test_case_handlers(options: dict,
                                 seed: Union[int, None],
                                 debug: bool = False,
                                 verbose: bool = False):
    # Unpack options
    ueg_opts = options['ueg_opts']
    nelec = options['nelec']
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    nsteps_per_block = options['nsteps_per_block']
    nblocks = options['nblocks']
    stabilize_freq = options['stabilize_freq']
    pop_control_freq = options['pop_control_freq']
    pop_control_method = options['pop_control_method']

    lowrank = options['lowrank']
    propagate = options.get('propagate', False)
    numpy.random.seed(seed)

    # 1. Generate UEG integrals.
    ueg = UEG(ueg_opts, verbose=verbose)
    ueg.build(verbose=verbose)
    nbasis = ueg.nbasis
    nchol = ueg.nchol
    nup, ndown = nelec

    if verbose:
        print(f"# nbasis = {nbasis}")
        print(f"# nchol = {nchol}")
        print(f"# nup = {nup}")
        print(f"# ndown = {ndown}")
        
    h1 = ueg.H1[0]
    chol = 2. * ueg.chol_vecs.toarray().copy()
    #ecore = ueg.ecore
    ecore = 0.

    # 2. Build Hamiltonian.
    hamiltonian = HamGeneric(
            numpy.array([h1, h1], dtype=numpy.complex128), 
            numpy.array(chol, dtype=numpy.complex128), 
            ecore,
            verbose=verbose)
    hamiltonian.name = options["hamiltonian"]["name"]
    hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    hamiltonian.sparse = options["hamiltonian"]["sparse"]

    # 3. Build trial.
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # 4. Build walkers.
    walkers = UHFThermalWalkers(trial, nbasis, nwalkers, lowrank=lowrank, 
                                verbose=verbose)
    
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


def build_driver_ueg_test_instance(options: Union[dict, None],
                                   seed: Union[int, None],
                                   debug: bool = False,
                                   verbose: bool = False):
    # Unpack options
    ueg_opts = options['ueg_opts']
    nelec = options['nelec']
    mu = options['mu']
    beta = options['beta']
    timestep = options['timestep']
    nwalkers = options['nwalkers']
    nsteps_per_block = options['nsteps_per_block']
    nblocks = options['nblocks']
    stabilize_freq = options['stabilize_freq']
    pop_control_freq = options['pop_control_freq']
    pop_control_method = options['pop_control_method']
    lowrank = options['lowrank']
    numpy.random.seed(seed)

    # 1. Generate UEG integrals.
    ueg = UEG(ueg_opts, verbose=verbose)
    ueg.build(verbose=verbose)
    nbasis = ueg.nbasis
    nchol = ueg.nchol
    nup, ndown = nelec

    if verbose:
        print(f"# nbasis = {nbasis}")
        print(f"# nchol = {nchol}")
        print(f"# nup = {nup}")
        print(f"# ndown = {ndown}")
        
    h1 = ueg.H1[0]
    chol = 2. * ueg.chol_vecs.toarray().copy()
    #ecore = ueg.ecore
    ecore = 0.

    # 2. Build Hamiltonian.
    hamiltonian = HamGeneric(
            numpy.array([h1, h1], dtype=numpy.complex128), 
            numpy.array(chol, dtype=numpy.complex128), 
            ecore,
            verbose=verbose)
    hamiltonian.name = options["hamiltonian"]["name"]
    hamiltonian._alt_convention = options["hamiltonian"]["_alt_convention"]
    hamiltonian.sparse = options["hamiltonian"]["sparse"]

    # 3. Build trial.
    trial = OneBody(hamiltonian, nelec, beta, timestep, verbose=verbose)

    # 4. Build Thermal AFQMC driver.
    afqmc = ThermalAFQMC.build(
                nelec, mu, beta, hamiltonian, trial, nwalkers, seed, 
                nblocks=nblocks, timestep=timestep, stabilize_freq=stabilize_freq,
                pop_control_freq=pop_control_freq, pop_control_method=pop_control_method,
                lowrank=lowrank, debug=debug, verbose=verbose)
    return afqmc

