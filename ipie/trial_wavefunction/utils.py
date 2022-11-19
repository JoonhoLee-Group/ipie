from ipie.trial_wavefunction.particle_hole import (
    ParticleHoleWicks,
    ParticleHoleWicksNonChunked,
)
from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.utils.io import (
    get_input_value,
    read_particle_hole_wavefunction,
    read_single_det_wavefunction,
    read_noci_wavefunction,
    determine_wavefunction_type,
)


def get_trial_wavefunction(
    system, hamiltonian, options={}, comm=None, scomm=None, verbose=False
):
    """Wavefunction factory.

    Parameters
    ----------
    system : class
        System class.
    hamiltonian : class
        Hamiltonian class.
    options : dict
        Trial wavefunction input options.
    comm : mpi communicator
        Global MPI communicator
    scomm : mpi communicator
        Shared communicator
    verbose : bool
        Print information.

    Returns
    -------
    trial : class or None
        Trial wavfunction class.
    """
    assert comm is not None
    if comm.rank == 0:
        if verbose:
            print("# Building trial wavefunction object.")
    wfn_file = get_input_value(
        options,
        "filename",
        default="wavefunction.h5",
        alias=["wavefunction_file"],
        verbose=verbose,
    )
    wfn_type = determine_wavefunction_type(wfn_file)
    ndets = get_input_value(options, "ndets", default=len(self.coeffs), verbose=verbose)
    ndets_props = get_input_value(
        options,
        "ndets_for_trial_props",
        default=min(ndets, 100),
        alias=["ndets_prop"],
        verbose=verbose,
    )
    if wfn_type == "particle_hole":
        wfn, phi0 = read_particle_hole_wavefunction(wfn_file)
        ndet_chunks = get_input_value(
            options,
            "ndet_chunks",
            default=1,
            alias=["nchunks", "chunks"],
            verbose=verbose,
        )
        if ndet_chunks == 1:
            trial = ParticleHoleWicksNonChunked(
                wfn,
                system.nelec,
                system.nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
            )
        else:
            trial = ParticleHoleWicks(
                wfn,
                system.nelec,
                system.nbasis,
                num_dets_for_trial=ndets,
                num_dets_for_props=ndets_props,
                num_det_chunks=ndet_chunks,
            )
    elif wfn_type == "noci":
        wfn, phi0 = read_noci_wavefunction(wfn_file)
        trial = NOCI(
            wfn,
            system.nelec,
            system.nbasis,
        )
        trial.num_dets = ndets
    else:
        wfn, phi0 = read_single_det_wavefunction(wfn_file)
        trial = SingleDet(
            wfn,
            system.nelec,
            system.nbasis,
        )
    trial.build()
    if verbose:
        print("# Number of determinants in trial wavefunction: {}".format(trial.num_dets))
    trial.half_rotate(system, hamiltonian, scomm)
