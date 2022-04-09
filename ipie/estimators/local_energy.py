import numpy
try:
    from ipie.legacy.estimators.ueg import local_energy_ueg
    from ipie.legacy.estimators.pw_fft import local_energy_pw_fft
except ImportError as e:
    print(e)
from ipie.legacy.estimators.hubbard import local_energy_hubbard, local_energy_hubbard_ghf,\
                                     local_energy_hubbard_holstein
from ipie.estimators.generic import (
    local_energy_generic_opt,
    local_energy_generic_cholesky_opt,
)
from ipie.legacy.estimators.generic import (
    local_energy_generic,
    local_energy_generic_pno,
    local_energy_generic_cholesky,
    local_energy_generic_cholesky_opt_stochastic
)
from ipie.legacy.estimators.thermal import particle_number, one_rdm_from_G
from ipie.legacy.estimators.ci import get_hmatel
from ipie.legacy.estimators.local_energy import local_energy_G as legacy_local_energy_G

def local_energy_G(system, hamiltonian, trial, G, Ghalf=None, X=None, Lap=None):
    assert len(G) == 2
    ghf = (G[0].shape[-1] == 2*hamiltonian.nbasis)
    # unfortunate interfacial problem for the HH model

    if hamiltonian.name == "Generic":
        if Ghalf is not None:
            if hamiltonian.exact_eri:
                return local_energy_generic_opt(system, G, Ghalf=Ghalf, eri=trial._eri)
            else:
                return local_energy_generic_cholesky_opt(system, hamiltonian, Ga = G[0], Gb= G[1],
                                                         Ghalfa=Ghalf[0],Ghalfb=Ghalf[1],
                                                         rchola=trial._rchola, rcholb=trial._rcholb)
        else:
            return local_energy_generic_cholesky(system, hamiltonian, G)
    else:
        return legacy_local_energy_G(system, hamiltonian, trial, G, Ghalf, X, Lap)

def local_energy(system, hamiltonian, walker, trial):
    return local_energy_G(system, hamiltonian, trial, walker.G, walker.Ghalf)
    
def variational_energy(system, hamiltonian, trial):
    if len(trial.psi.shape) == 2 or len(trial.psi) == 1:
        return local_energy(system, hamiltonian, trial, trial)
    else:
        print("# MSD trial disabled at the moment")
        exit(1)

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