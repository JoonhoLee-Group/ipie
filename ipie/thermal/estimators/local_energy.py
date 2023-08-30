import numpy

try:
    from ipie.thermal.estimators.pw_fft import local_energy_pw_fft
    from ipie.thermal.estimators.ueg import local_energy_ueg
except ImportError as e:
    print(e)
from ipie.estimators.generic import local_energy_generic_opt
from ipie.thermal.estimators.generic import (
    local_energy_generic_cholesky,
    local_energy_generic_cholesky_opt,
    local_energy_generic_cholesky_opt_stochastic,
    local_energy_generic_pno,
)
from ipie.thermal.estimators.hubbard import (
    local_energy_hubbard,
    local_energy_hubbard_ghf,
    local_energy_hubbard_holstein,
)
from ipie.thermal.estimators.thermal import one_rdm_from_G


def local_energy_G(system, hamiltonian, trial, G, Ghalf=None, X=None, Lap=None):
    assert len(G) == 2
    ghf = G[0].shape[-1] == 2 * hamiltonian.nbasis
    # unfortunate interfacial problem for the HH model
    if hamiltonian.name == "Hubbard":
        if ghf:
            return local_energy_hubbard_ghf(hamiltonian, G)
        else:
            return local_energy_hubbard(hamiltonian, G)
    elif hamiltonian.name == "HubbardHolstein":
        return local_energy_hubbard_holstein(hamiltonian, G, X, Lap, Ghalf)
    elif hamiltonian.name == "PW_FFT":
        return local_energy_pw_fft(system, G, Ghalf)
    elif hamiltonian.name == "UEG":
        return local_energy_ueg(system, hamiltonian, G)
    else:
        if Ghalf is not None:
            if hamiltonian.stochastic_ri and hamiltonian.control_variate:
                return local_energy_generic_cholesky_opt_stochastic(
                    system,
                    G,
                    nsamples=hamiltonian.nsamples,
                    Ghalf=Ghalf,
                    rchol=trial._rchol,
                    C0=trial.psi,
                    ecoul0=trial.ecoul0,
                    exxa0=trial.exxa0,
                    exxb0=trial.exxb0,
                )
            elif hamiltonian.stochastic_ri and not hamiltonian.control_variate:
                return local_energy_generic_cholesky_opt_stochastic(
                    system,
                    G,
                    nsamples=hamiltonian.nsamples,
                    Ghalf=Ghalf,
                    rchol=trial._rchol,
                )
            elif hamiltonian.exact_eri and not hamiltonian.pno:
                return local_energy_generic_opt(system, G, Ghalf=Ghalf, eri=trial._eri)
            elif hamiltonian.pno:
                assert hamiltonian.exact_eri and hamiltonian.control_variate
                return local_energy_generic_pno(
                    system,
                    G,
                    Ghalf=Ghalf,
                    eri=trial._eri,
                    C0=trial.C0,
                    ecoul0=trial.ecoul0,
                    exxa0=trial.exxa0,
                    exxb0=trial.exxb0,
                    UVT=trial.UVT,
                )
            else:
                return local_energy_generic_cholesky_opt(
                    system,
                    hamiltonian,
                    Ga=G[0],
                    Gb=G[1],
                    Ghalfa=Ghalf[0],
                    Ghalfb=Ghalf[1],
                    rchola=trial._rchola,
                    rcholb=trial._rcholb,
                )
        else:
            return local_energy_generic_cholesky(system, hamiltonian, G)


# TODO: should pass hamiltonian here and make it work for all possible types
# this is a generic local_energy handler. So many possible combinations of local energy strategies...
def local_energy(system, hamiltonian, walker, trial):
    if walker.name == "MultiDetWalker":
        if hamiltonian.name == "HubbardHolstein":
            return local_energy_multi_det_hh(
                system, walker.Gi, walker.weights, walker.X, walker.Lapi
            )
        else:
            return local_energy_multi_det(system, hamiltonian, trial, walker.Gi, walker.weights)
    elif walker.name == "ThermalWalker":
        return local_energy_G(system, hamiltonian, trial, one_rdm_from_G(walker.G), None)
    else:
        if hamiltonian.name == "HubbardHolstein":
            return local_energy_G(
                system, hamiltonian, trial, walker.G, walker.Ghalf, walker.X, walker.Lap
            )
        else:
            return local_energy_G(system, hamiltonian, trial, walker.G, walker.Ghalf)


def local_energy_multi_det(system, hamiltonian, trial, Gi, weights):
    weight = 0
    energies = 0
    denom = 0
    for idet, (w, G) in enumerate(zip(weights, Gi)):
        energies += w * numpy.array(local_energy_G(system, hamiltonian, trial, G))
        denom += w
    return tuple(energies / denom)


def local_energy_multi_det_hh(system, Gi, weights, X, Lapi):
    weight = 0
    energies = 0
    denom = 0
    for w, G, Lap in zip(weights, Gi, Lapi):
        # construct "local" green's functions for each component of A
        energies += w * numpy.array(local_energy_hubbard_holstein(system, G, X, Lap, Ghalf=None))
        denom += w
    return tuple(energies / denom)


