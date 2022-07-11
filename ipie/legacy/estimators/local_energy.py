import numpy

try:
    from ipie.legacy.estimators.pw_fft import local_energy_pw_fft
    from ipie.legacy.estimators.ueg import local_energy_ueg
except ImportError as e:
    print(e)
from ipie.estimators.generic import local_energy_generic_opt
from ipie.legacy.estimators.ci import get_hmatel
from ipie.legacy.estimators.generic import (
    local_energy_generic, local_energy_generic_cholesky,
    local_energy_generic_cholesky_opt,
    local_energy_generic_cholesky_opt_stochastic, local_energy_generic_pno)
from ipie.legacy.estimators.hubbard import (local_energy_hubbard,
                                            local_energy_hubbard_ghf,
                                            local_energy_hubbard_holstein)
from ipie.legacy.estimators.thermal import one_rdm_from_G, particle_number


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
            return local_energy_multi_det(
                system, hamiltonian, trial, walker.Gi, walker.weights
            )
    elif walker.name == "ThermalWalker":
        return local_energy_G(
            system, hamiltonian, trial, one_rdm_from_G(walker.G), None
        )
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
        energies += w * numpy.array(
            local_energy_hubbard_holstein(system, G, X, Lap, Ghalf=None)
        )
        denom += w
    return tuple(energies / denom)


# def local_energy(system, hamiltonian, walker, trial):
def variational_energy(system, hamiltonian, trial):
    if len(trial.psi.shape) == 2 or len(trial.psi) == 1:
        return local_energy(system, hamiltonian, trial, trial)
    else:
        print("# MSD trial disabled at the moment")
        exit(1)


def variational_energy_multi_det(system, psi, coeffs, H=None, S=None):
    weight = 0
    energies = 0
    denom = 0
    nup = system.nup
    ndet = len(coeffs)
    if H is not None and S is not None:
        store = True
    else:
        store = False
    for i, (Bi, ci) in enumerate(zip(psi, coeffs)):
        for j, (Aj, cj) in enumerate(zip(psi, coeffs)):
            # construct "local" green's functions for each component of A
            Gup, GHup, inv_O_up = gab_mod_ovlp(Bi[:, :nup], Aj[:, :nup])
            Gdn, GHdn, inv_O_dn = gab_mod_ovlp(Bi[:, nup:], Aj[:, nup:])
            ovlp = 1.0 / (scipy.linalg.det(inv_O_up) * scipy.linalg.det(inv_O_dn))
            weight = (ci.conj() * cj) * ovlp
            G = numpy.array([Gup, Gdn])
            e = numpy.array(local_energy(system, G))
            if store:
                H[i, j] = ovlp * e[0]
                S[i, j] = ovlp
            energies += weight * e
            denom += weight
    return tuple(energies / denom)


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
        denom += ci.conj() * ci
        for j in range(0, i + 1):
            cj = coeffs[j]
            occj = occs[j]
            etot, e1b, e2b = ci.conj() * cj * get_hmatel(ham, nel, occi, occj)
            evar += etot
            one_body += e1b
            two_body += e2b
            if j < i:
                # Use Hermiticity
                evar += etot
                one_body += e1b
                two_body += e2b
    return evar / denom, one_body / denom, two_body / denom


def variational_energy_single_det(
    system,
    psi,
    G=None,
    GH=None,
    rchol=None,
    eri=None,
    C0=None,
    ecoul0=None,
    exxa0=None,
    exxb0=None,
    UVT=None,
):
    assert len(psi.shape) == 2
    return local_energy(
        system,
        G,
        Ghalf=GH,
        rchol=rchol,
        eri=eri,
        C0=C0,
        ecoul0=ecoul0,
        exxa0=exxa0,
        exxb0=exxb0,
        UVT=UVT,
    )
