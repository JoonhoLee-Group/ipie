from cuquantum import cutensornet, NetworkOptions, contract
from ipie.utils.backend import arraylib as xp

def slice_g_kpq_k_given_q(gf, iq, kpq_mat):
    """
    slice the Green's function G^{w}_{pk+q, rk} to g^{w}_{kpr} for a given q
    """
    nw = gf.shape[0]
    nk = gf.shape[1]
    nocc = gf.shape[2]
    nbsf = gf.shape[-1]
    kpq = kpq_mat[:, iq][None, :, None, None]
    w = xp.arange(nw)[:, None, None, None]
    k = xp.arange(nk)[None, :, None, None]
    i = xp.arange(nocc)[None, None, :, None]
    p = xp.arange(nbsf)[None, None, None, :]
    g_kpq = gf[w, kpq, i, k, p]
    return g_kpq

def slice_g_k_kpq_given_q(gf, iq, kpq_mat):
    """
    slice the Green's function G^{w}_{pk+q, rk} to g^{w}_{kpr} for a given q
    """
    nw = gf.shape[0]
    nk = gf.shape[1]
    nocc = gf.shape[2]
    nbsf = gf.shape[-1]
    kpq = kpq_mat[:, iq][None, :, None, None]
    w = xp.arange(nw)[:, None, None, None]
    k = xp.arange(nk)[None, :, None, None]
    i = xp.arange(nocc)[None, None, :, None]
    p = xp.arange(nbsf)[None, None, None, :]
    g_kpq = gf[w, k, i, kpq, p]
    return g_kpq

def contract_gf_cgto_kpq_k(gf, cgto, iq_real, kpq_mat):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    nk = cgto.shape[0]
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    ikpq = kpq_mat[iq_real]
    cgto_kpq = cgto[ikpq]
    g_kpq = slice_g_kpq_k_given_q(gf, iq_real, kpq_mat)
    out_q = contract("kPp, kPr, wkpr -> wP", cgto_kpq.conj(), cgto, g_kpq, options=network_opts)
    cutensornet.destroy(handle)
    return out_q

def contract_gf_cgto_kmq_k(gf, cgto, iq_real, kmq_mat):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    nk = cgto.shape[0]
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    ikmq = kmq_mat[:, iq_real]
    cgto_kmq = cgto[ikmq]
    g_kpq = slice_g_kpq_k_given_q(gf, iq_real, kmq_mat)
    out_q = contract("kPp, kPr, wkpr -> wP", cgto_kmq.conj(), cgto, g_kpq, options=network_opts)
    cutensornet.destroy(handle)
    return out_q

def contract_gf_cgto12_kpq_k(gf, cgto1, cgto2, iq_real, kpq_mat):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    nk = cgto1.shape[0]
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    ikpq = kpq_mat[iq_real]
    cgto1_kpq = cgto1[ikpq]
    g_kpq = slice_g_kpq_k_given_q(gf, iq_real, kpq_mat)
    out_q = contract("kPp, kPr, wkpr -> wP", cgto1_kpq.conj(), cgto2, g_kpq, options=network_opts)
    cutensornet.destroy(handle)
    return out_q


def contract_gf_cgto12_kmq_k(gf, cgto1, cgto2, iq_real, kmq_mat):
    """
    perform the contraction: psi^{k+q}_{pP}.conj(), psi^{k}_{rP}, G^{w}_{pk+q, rk} -> X^w_{Pq}
    """
    nk = cgto1.shape[0]
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    ikmq = kmq_mat[:, iq_real]
    cgto1_kmq = cgto1[ikmq]
    g_kmq = slice_g_kpq_k_given_q(gf, iq_real, kmq_mat)
    out_q = contract("kPp, kPr, wkpr -> wP", cgto1_kmq.conj(), cgto2, g_kmq, options=network_opts)
    cutensornet.destroy(handle)
    return out_q

def contract_gf_cgto_k_kpq(gf, cgto, iq_real, kpq_mat):
    """
    perform the contraction: psi^{k}_{pP}.conj(), psi^{k+q}_{rP}, G^{w}_{pk, rk+q} -> X^w_{Pq}
    """
    nk = cgto.shape[0]
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    ikpq = kpq_mat[iq_real]
    cgto_kpq = cgto[ikpq]
    g_kpq = slice_g_k_kpq_given_q(gf, iq_real, kpq_mat)
    out_q = contract("kPp, kPr, wkpr -> wP", cgto_kpq.conj(), cgto, g_kpq, options=network_opts)
    cutensornet.destroy(handle)
    return out_q

def contract_gf_cgto_k_kmq(gf, cgto, iq_real, kmq_mat):
    """
    perform the contraction: psi^{k}_{pP}.conj(), psi^{k-q}_{rP}, G^{w}_{pk, rk-q} -> X^w_{Pq}
    """
    nk = cgto.shape[0]
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    ikmq = kmq_mat[:, iq_real]
    cgto_kmq = cgto[ikmq]
    g_kpq = slice_g_k_kpq_given_q(gf, iq_real, kmq_mat)
    out_q = contract("kPp, kPr, wkpr -> wP", cgto_kmq.conj(), cgto, g_kpq, options=network_opts)
    cutensornet.destroy(handle)
    return out_q

def contract_gf_cgto12_k_kpq(gf, cgto1, cgto2, iq_real, kpq_mat):
    """
    perform the contraction: psi1^{k}_{pP}.conj(), psi2^{k+q}_{rP}, G^{w}_{pk, rk+q} -> X^w_{Pq}
    """
    nk = cgto1.shape[0]
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    ikpq = kpq_mat[iq_real]
    cgto1_kpq = cgto1[ikpq]
    g_kpq = slice_g_k_kpq_given_q(gf, iq_real, kpq_mat)
    out_q = contract("kPp, kPr, wkpr -> wP", cgto1_kpq.conj(), cgto2, g_kpq, options=network_opts)
    cutensornet.destroy(handle)
    return out_q


def contract_gf_cgto12_k_kmq(gf, cgto1, cgto2, iq_real, kmq_mat):
    """
    perform the contraction: psi1^{k}_{pP}.conj(), psi2^{k-q}_{rP}, G^{w}_{pk, rk-q} -> X^w_{Pq}
    """
    nk = cgto1.shape[0]
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    ikmq = kmq_mat[:, iq_real]
    cgto1_kmq = cgto1[ikmq]
    g_kmq = slice_g_k_kpq_given_q(gf, iq_real, kmq_mat)
    out_q = contract("kPp, kPr, wkpr -> wP", cgto1_kmq.conj(), cgto2, g_kmq, options=network_opts)
    cutensornet.destroy(handle)
    return out_q