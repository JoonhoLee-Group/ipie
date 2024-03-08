from ipie.addons.thermal.trial.mean_field import MeanField
from ipie.addons.thermal.trial.one_body import OneBody


def get_trial_density_matrix(hamiltonian, nelec, beta, dt, options={}, comm=None, verbose=False):
    """Wrapper to select trial wavefunction class.

    Parameters
    ----------

    Returns
    -------
    trial : class or None
        Trial density matrix class.
    """
    trial_type = options.get("name", "one_body")
    if comm is None or comm.rank == 0:
        if trial_type == "one_body_mod":
            trial = OneBody(
                hamiltonian,
                nelec,
                beta,
                dt,
                options=options,
                H1=hamiltonian.h1e_mod,
                verbose=verbose,
            )

        elif trial_type == "one_body":
            trial = OneBody(hamiltonian, nelec, beta, dt, options=options, verbose=verbose)

        elif trial_type == "thermal_hartree_fock":
            trial = MeanField(hamiltonian, nelec, beta, dt, options=options, verbose=verbose)

        else:
            trial = None

    else:
        trial = None

    if comm is not None:
        trial = comm.bcast(trial)

    return trial
