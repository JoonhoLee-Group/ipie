from cuquantum import cutensornet, NetworkOptions, contract
from ipie.utils.backend import arraylib as xp

def slice_gf_kpq_k_given_q(gf, iq, kpq_mat):
    """
    slice the Green's function G^{w}_{pk+q, rk} to g^{w}_{kpr} for a given q
    """
    nw = gf.shape[0]
    nk = gf.shape[1]
    nocc = gf.shape[2]
    nbsf = gf.shape[-1]
    kpq = kpq_mat[iq][None, :, None, None]
    w = xp.arange(nw)[:, None, None, None]
    k = xp.arange(nk)[None, :, None, None]
    i = xp.arange(nocc)[None, None, :, None]
    p = xp.arange(nbsf)[None, None, None, :]
    gf_kpq = gf[w, kpq, i, k, p]
    return gf_kpq

def slice_gf_k_kpq_given_q(gf, iq, kpq_mat):
    """
    slice the Green's function G^{w}_{pk, rk+q} to g^{w}_{kpr} for a given q
    """
    nw = gf.shape[0]
    nk = gf.shape[1]
    nocc = gf.shape[2]
    nbsf = gf.shape[-1]
    kpq = kpq_mat[iq][None, :, None, None]
    w = xp.arange(nw)[:, None, None, None]
    k = xp.arange(nk)[None, :, None, None]
    i = xp.arange(nocc)[None, None, :, None]
    p = xp.arange(nbsf)[None, None, None, :]
    gf_kpq = gf[w, k, i, kpq, p]
    return gf_kpq

def contract_gf_cgto_kpq_k(gf_kpq, cgto, cgto_kpq, iq_real, network_opts):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    out_q = contract("kPp, kPr, wkpr -> wP", cgto_kpq.conj(), cgto, gf_kpq, options=network_opts)
    return out_q

def contract_gf_cgto_kmq_k(gf_kmq, cgto, cgto_kmq, iq_real, network_opts):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    out_q = contract("kPp, kPr, wkpr -> wP", cgto_kmq.conj(), cgto, gf_kpq, options=network_opts)
    return out_q

def contract_gf_cgto12_kpq_k(gf_kpq, cgto1_kpq, cgto2, iq_real, network_opts):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    out_q = contract("kPp, kPr, wkpr -> wP", cgto1_kpq.conj(), cgto2, gf_kpq, options=network_opts)
    return out_q


def contract_gf_cgto12_kmq_k(gf_kmq, cgto1_kmq, cgto2, iq_real, network_opts):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    out_q = contract("kPp, kPr, wkpr -> wP", cgto1_kmq.conj(), cgto2, gf_kmq, options=network_opts)
    return out_q

def contract_gf_cgto_k_kpq(gf_kpq, cgto, cgto_kpq, iq_real, network_opts):
    """
    perform the contraction: psi^{k}_{pP}.conj(), psi^{k+q}_{rP}, G^{w}_{pk, rk+q} -> X^w_{Pq}
    """
    out_q = contract("kPp, kPr, wkpr -> wP", cgto.conj(), cgto_kpq, gf_kpq, options=network_opts)
    return out_q

def contract_gf_cgto_k_kmq(gf_kmq, cgto, cgto_kmq, iq_real, network_opts):
    """
    perform the contraction: psi^{k}_{pP}.conj(), psi^{k-q}_{rP}, G^{w}_{pk, rk-q} -> X^w_{Pq}
    """
    out_q = contract("kPp, kPr, wkpr -> wP", cgto_kmq.conj(), cgto, gf_kpq, options=network_opts)
    return out_q

def contract_gf_cgto12_k_kpq(gf_kpq, cgto1, cgto2_kpq, iq_real, network_opts):
    """
    perform the contraction: psi1^{k}_{pP}.conj(), psi2^{k+q}_{rP}, G^{w}_{pk, rk+q} -> X^w_{Pq}
    """
    out_q = contract("kPp, kPr, wkpr -> wP", cgto1.conj(), cgto2_kpq, gf_kpq, options=network_opts)
    return out_q


def contract_gf_cgto12_k_kmq(gf_kmq, cgto1, cgto2_kmq, iq_real, network_opts):
    """
    perform the contraction: psi1^{k}_{pP}.conj(), psi2^{k-q}_{rP}, G^{w}_{pk, rk-q} -> X^w_{Pq}
    """
    out_q = contract("kPp, kPr, wkpr -> wP", cgto1.conj(), cgto2_kmq, gf_kmq, options=network_opts)
    return out_q