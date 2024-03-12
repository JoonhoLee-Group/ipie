from ipie.addons.thermal.estimators.generic import local_energy_generic_cholesky
from ipie.addons.thermal.estimators.thermal import one_rdm_from_G

def local_energy_P(hamiltonian, trial, P):
    """Compute local energy from a given density matrix P.

    Parameters
    ----------
    hamiltonian : hamiltonian object
        Hamiltonian being studied.
    trial : trial wavefunction object
        Trial wavefunction.
    P : np.ndarray
        Walker density matrix.

    Returns:
    -------
    local_energy : tuple / array
        Total, one-body and two-body energies.
    """
    assert len(P) == 2
    return local_energy_generic_cholesky(hamiltonian, P)

def local_energy(hamiltonian, walker, trial):
    return local_energy_P(hamiltonian, trial, one_rdm_from_G(walker.G))
