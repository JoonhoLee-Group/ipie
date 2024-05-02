import torch


def get_local_e(rh1, Ghalf, rchol, nuc):
    """Get the local energy using the rotated cholesky vectors and the green's function
    Parameters
    ----------
    hcore : 2d array with size (nao, nao), the unmodified hcore

    Returns
    -------
    local_e : 1d array with size (nwalkers)
    """
    local_e1 = 2 * (
        torch.einsum("ij, aji->a", rh1, Ghalf.real)
        + 1j * torch.einsum("ij, aji->a", rh1, Ghalf.imag)
    )
    local_e2 = 0.5 * (get_Coulomb(rchol, Ghalf) - get_Exchange(rchol, Ghalf))
    local_nuc = nuc
    local_e = local_nuc + local_e1 + local_e2
    return local_e


def get_trial_e(rh1, G, rchol, nuc):
    """Get the local energy of trial wavefunction
    Parameters
    ----------
    hcore : 2d array with size (nao, nao), the unmodified hcore

    Returns
    -------
    local_e : 1d array with size (nwalkers)
    """
    local_e1 = 2 * torch.einsum("ij, ji->", rh1, G)
    local_e2 = 0.5 * (get_Coulomb_trial(rchol, G) - get_Exchange_trial(rchol, G))
    local_nuc = nuc
    local_e = local_nuc + local_e1 + local_e2
    return local_e


def get_Coulomb(rchol, Ghalf):
    """
    E_C = \sum_{i,p,j,q, g, n} L^\gamma_{ip} G^{n}_{ip} L^\gamma_{jq} G^{n}_{jq}
    input:
        rchol, 3d tensor with size (nwalkers, nocc, nao), the cholesky decomposition of the 2-body hamiltonian
        ghalf, 3d tensor with size (nwalkers, nocc, nao), the green's function of the trial wavefunction
    """
    X = 2 * (
        torch.einsum("pij, aji -> ap", rchol, Ghalf.real)
        + 1j * torch.einsum("pij, aji->ap", rchol, Ghalf.imag)
    )
    EJ = torch.einsum("ap, ap -> a", X, X)
    return EJ


def get_Exchange(rchol, Ghalf):
    """
    E_X = \sum_{i,p,j,q, g, n} L^\gamma_{ip} G^{n}_{jq} L^\gamma_{jq} G^{n}_{ip}
    input:
        rchol, 3d tensor with size (nwalkers, nocc, nao), the cholesky decomposition of the 2-body hamiltonian
        ghalf, 3d tensor with size (nwalkers, nocc, nao), the green's function of the trial wavefunction
    """
    EXreal = 2 * (
        torch.einsum("gip, apj, gjq, aqi -> a", rchol, Ghalf.real, rchol, Ghalf.real)
        - torch.einsum("gip, apj, gjq, aqi -> a", rchol, Ghalf.imag, rchol, Ghalf.imag)
    )
    EXimag = (
        1j
        * 2
        * (
            torch.einsum("gip, apj, gjq, aqi -> a", rchol, Ghalf.real, rchol, Ghalf.imag)
            + torch.einsum("gip, apj, gjq, aqi -> a", rchol, Ghalf.imag, rchol, Ghalf.real)
        )
    )
    EX = EXreal + EXimag
    return EX


def get_Coulomb_trial(rchol, G):
    """
    E_C = \sum_{i,p,j,q, g} L^\gamma_{ip} G_{ip} L^\gamma_{jq} G_{jq}
    input:
        rchol, 3d tensor with size (nwalkers, nocc, nao), the cholesky decomposition of the 2-body hamiltonian
        ghalf, 3d tensor with size (nwalkers, nocc, nao), the green's function of the trial wavefunction
    """
    X = 2 * torch.einsum("pij, ji -> p", rchol, G)
    EJ = torch.einsum("p, p -> ", X, X)
    return EJ


def get_Exchange_trial(rchol, G):
    """
    E_X = \sum_{i,p,j,q, g} L^\gamma_{ip} G_{jq} L^\gamma_{jq} G_{ip}
    input:
        rchol, 3d tensor with size (nwalkers, nocc, nao), the cholesky decomposition of the 2-body hamiltonian
        ghalf, 3d tensor with size (nwalkers, nocc, nao), the green's function of the trial wavefunction
    """
    EX = 2 * torch.einsum("gip, pj, gjq, qi ->", rchol, G, rchol, G)
    return EX
