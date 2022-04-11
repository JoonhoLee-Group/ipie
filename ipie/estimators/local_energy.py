import numpy
from ipie.legacy.estimators.ci import get_hmatel
from ipie.legacy.estimators.local_energy import local_energy_G as legacy_local_energy_G
from ipie.estimators.generic import (
    local_energy_generic_opt,
    half_rotated_cholesky_dG,
    local_energy_generic_cholesky
)

def local_energy_G(system, hamiltonian, trial, G, Ghalf):
    assert len(G) == 2
    ghf = (G[0].shape[-1] == 2*hamiltonian.nbasis)
    # unfortunate interfacial problem for the HH model

    if hamiltonian.name == "Generic":
        if Ghalf is not None:
            if hamiltonian.exact_eri:
                return local_energy_generic_opt(system, G, Ghalf=Ghalf, eri=trial._eri)
            else:
                # return local_energy_generic_cholesky_opt(system, hamiltonian.ecore, Ghalfa=Ghalf[0], Ghalfb=Ghalf[1], trial=trial)
                return half_rotated_cholesky_dG(system, hamiltonian.ecore, Ghalfa=Ghalf[0], Ghalfb=Ghalf[1], trial=trial)
        else:
            return local_energy_generic_cholesky(system, hamiltonian, G)
    else:
        return legacy_local_energy_G(system, hamiltonian, trial, G, Ghalf)

def local_energy(system, hamiltonian, walker, trial):
    return local_energy_G(system, hamiltonian, trial, walker.G, walker.Ghalf)
    
def variational_energy(system, hamiltonian, trial):
    assert (len(trial.psi.shape) == 2 or len(trial.psi) == 1)
    return local_energy(system, hamiltonian, trial, trial)

def variational_energy_ortho_det(system, ham, occs, coeffs):
    """Compute variational energy for CI-like multi-determinant expansion.

    Parameters
    ----------
    system : :class:`ipie.system` object
        System object.
    occs : list of lists
        list of determinants.
    coeffs : :class:`numpy.ndarray`
        Expansion coefficients.

    Returns
    -------
    energy : tuple of float / complex
        Total energies: (etot,e1b,e2b).
    """
    evar = 0.0
    denom = 0.0
    one_body = 0.0
    two_body = 0.0
    nel = system.nup + system.ndown
    for i, (occi, ci) in enumerate(zip(occs, coeffs)):
        denom += ci.conj()*ci
        for j in range(0,i+1):
            cj = coeffs[j]
            occj = occs[j]
            etot, e1b, e2b = ci.conj()*cj*get_hmatel(ham, nel, occi, occj)
            evar += etot
            one_body += e1b
            two_body += e2b
            if j < i:
                # Use Hermiticity
                evar += etot
                one_body += e1b
                two_body += e2b
    return evar/denom, one_body/denom, two_body/denom