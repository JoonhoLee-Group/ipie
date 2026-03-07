# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from math import ceil

import numpy
from numba import jit

from ipie.config import config
from ipie.utils.backend import arraylib as xp
from ipie.utils.contract_gf_cgto import (
    contract_gf_cgto12_kpq_k,
    contract_gf_cgto12_k_kpq,
    slice_gf_k_kpq_given_q,
    slice_gf_kpq_k_given_q,
)
from ipie.utils.cuquantum_backend import (
    NetworkOptions_required as NetworkOptions,
    contract_required as contract,
    cutensornet_required as cutensornet,
)


def kpt_isdf_ecoul_kernel_gpu(
    MPQ, halfrot_cgtoa, halfrot_cgtob, cgto, Ghalfa_batch, Ghalfb_batch, kpq_mat, Sset, Qplus
):
    nk = cgto.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    ecoul = xp.zeros(nwalkers, dtype=numpy.complex128)
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        rcgtob_kpq = halfrot_cgtob[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_k_kpq = slice_gf_k_kpq_given_q(Ghalfb_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(
            ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real, network_opts
        ) + contract_gf_cgto12_k_kpq(gb_k_kpq, halfrot_cgtob, cgto_kpq, iq_real, network_opts)
        del ga_k_kpq
        del gb_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_kpq_k = slice_gf_kpq_k_given_q(Ghalfb_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(
            ga_kpq_k, rcgtoa_kpq, cgto, iq_real, network_opts
        ) + contract_gf_cgto12_kpq_k(gb_kpq_k, rcgtob_kpq, cgto, iq_real, network_opts)
        del ga_kpq_k
        del gb_kpq_k
        ecoul += xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        rcgtob_kpq = halfrot_cgtob[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_k_kpq = slice_gf_k_kpq_given_q(Ghalfb_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(
            ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real, network_opts
        ) + contract_gf_cgto12_k_kpq(gb_k_kpq, halfrot_cgtob, cgto_kpq, iq_real, network_opts)
        del ga_k_kpq
        del gb_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        gb_kpq_k = slice_gf_kpq_k_given_q(Ghalfb_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(
            ga_kpq_k, rcgtoa_kpq, cgto, iq_real, network_opts
        ) + contract_gf_cgto12_kpq_k(gb_kpq_k, rcgtob_kpq, cgto, iq_real, network_opts)
        del ga_kpq_k
        del gb_kpq_k
        ecoul += 2.0 * xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)
    cutensornet.destroy(handle)
    return 0.5 * ecoul / nk


def kpt_isdf_ecoul_rhf_kernel_gpu(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus):
    nk = cgto.shape[0]
    nwalkers = Ghalfa_batch.shape[0]
    ecoul = xp.zeros(nwalkers, dtype=numpy.complex128)
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    for iq in range(len(Sset)):
        iq_real = Sset[iq]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real, network_opts)
        del ga_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(ga_kpq_k, rcgtoa_kpq, cgto, iq_real, network_opts)
        del ga_kpq_k
        ecoul += xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)

    for iq in range(len(Sset), len(Sset) + len(Qplus)):
        iq_real = Qplus[iq - len(Sset)]
        MPQ_iq = MPQ[iq]
        ikpq = kpq_mat[iq_real]
        cgto_kpq = cgto[ikpq]
        rcgtoa_kpq = halfrot_cgtoa[ikpq]
        ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v1_wP = contract_gf_cgto12_k_kpq(ga_k_kpq, halfrot_cgtoa, cgto_kpq, iq_real, network_opts)
        del ga_k_kpq
        ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_batch, iq_real, kpq_mat)
        v2_wP = contract_gf_cgto12_kpq_k(ga_kpq_k, rcgtoa_kpq, cgto, iq_real, network_opts)
        del ga_kpq_k
        ecoul += 2.0 * xp.sum((v1_wP @ MPQ_iq) * v2_wP, axis=1)
    cutensornet.destroy(handle)
    return 2.0 * ecoul / nk


def contract_psi_G_psi(psiiP_slice, psiqQ_slice, G, buff, nP, nQ):
    nw, nk, nocc, nbsf = G.shape[0], G.shape[1], G.shape[2], G.shape[4]
    G = G.transpose(3, 0, 1, 2, 4).reshape(nk, nw * nk * nocc, nbsf)
    size_Gpsi = nw * nocc * nk**2 * nQ
    Gpsi = buff[:size_Gpsi].reshape(nk, nw * nocc * nk, nQ)
    xp.matmul(G, psiqQ_slice, out=Gpsi)
    Gpsi = (
        Gpsi.reshape(nk, nw, nk, nocc, nQ).transpose(2, 1, 4, 0, 3).reshape(nk, nw * nQ * nk, nocc)
    )
    size_TPQ = nk**2 * nw * nP * nQ
    TPQ = buff[:size_TPQ].reshape(nk, nw * nQ * nk, nP)
    xp.matmul(Gpsi, psiiP_slice, out=TPQ)
    return TPQ


def X_contract_cupy_lowk(
    halfrot_cgtoa,
    phikr_kpq,
    M_PQ_iq,
    phiki_kpq,
    cgto,
    Ga_chunk,
    G_kpq_kprimepq_chunk,
    buff1,
    buff2,
    slices_isdf,
):
    nw = Ga_chunk.shape[0]
    nk = cgto.shape[0]
    exx = xp.zeros((nw,), dtype=xp.complex128)

    for P in slices_isdf:
        psi_iP_k = halfrot_cgtoa[:, P, :].transpose(0, 2, 1).conj()
        phip_kpq_P = phikr_kpq[:, P, :].transpose(0, 2, 1)
        nP = P.stop - P.start
        for Q in slices_isdf:
            nQ = Q.stop - Q.start
            psi_iQ_kp = cgto[:, Q, :].transpose(0, 2, 1)
            phij_kpq_Q = phiki_kpq[:, Q, :].transpose(0, 2, 1).conj()
            TPQ = contract_psi_G_psi(psi_iP_k, psi_iQ_kp, Ga_chunk, buff1, nP, nQ)
            TQP_kpq = contract_psi_G_psi(
                phij_kpq_Q, phip_kpq_P, G_kpq_kprimepq_chunk, buff2, nQ, nP
            )
            TPQ = (
                TPQ.reshape(nk, nw, nQ, nk, nP)
                .transpose(1, 0, 3, 4, 2)
                .reshape(nw, nk * nk, nP, nQ)
            )
            TQP_kpq = (
                TQP_kpq.reshape(nk, nw, nP, nk, nQ)
                .transpose(1, 3, 0, 2, 4)
                .reshape(nw, nk * nk, nP, nQ)
            )
            Tsq = xp.sum(TPQ * TQP_kpq, axis=1)
            M_PQ_iq_sliced = M_PQ_iq[P, Q].astype(xp.complex128, copy=False)
            exx += Tsq.reshape(nw, nP * nQ) @ M_PQ_iq_sliced.ravel()
    return exx


def kpt_isdf_exx_kernel_gpu(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch, kpq_mat, Sset, Qplus):
    nwalker, nk, nocc, _, nbsf = Ghalfa_batch.shape
    nisdf = MPQ.shape[-1]
    if nbsf > 8 * nk:
        exx = xp.zeros((nwalker,), dtype=xp.complex128)

        w_idx = xp.arange(nwalker)[:, None, None, None, None]
        k_idx = xp.arange(nk)[None, :, None, None, None]
        i_idx = xp.arange(nocc)[None, None, :, None, None]
        kprime_idx = xp.arange(nk)[None, None, None, :, None]
        p_idx = xp.arange(nbsf)[None, None, None, None, :]

        intermediate_mem = nisdf * nisdf * nk**2 * nwalker * 3 * 16 / 1024**3
        max_mem = 8.0
        num_chunks = ceil(intermediate_mem / max_mem)
        num_chunk_per_dim = ceil(num_chunks**0.5)
        nisdf_per_chunk = ceil(nisdf / num_chunk_per_dim)
        nisdf_left = nisdf
        slices_isdf = []
        for i_chunk in range(num_chunks):
            if nisdf_left == 0:
                break
            nisdf_chunk = min(nisdf_left, nisdf_per_chunk)
            nisdf_left -= nisdf_chunk
            slices_isdf.append(
                slice(i_chunk * nisdf_per_chunk, i_chunk * nisdf_per_chunk + nisdf_chunk)
            )
        max_dim = max(nisdf_per_chunk, nocc)
        buff1 = xp.empty(nwalker * nk**2 * max_dim**2, dtype=xp.complex128)
        buff2 = xp.empty(nwalker * nk**2 * max_dim**2, dtype=xp.complex128)
        for iq in range(len(Sset)):
            iq_real = Sset[iq]
            ikpq = kpq_mat[iq_real]
            phikr_kpq = cgto[ikpq]
            phiki_kpq = halfrot_cgtoa[ikpq]
            kpq_idx = kpq_mat[k_idx, iq_real]
            kprimepq_idx = kpq_mat[kprime_idx, iq_real]
            G_kpq_kprimepq_chunk = Ghalfa_batch[w_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
            MPQ_iq = MPQ[iq]
            exx_iq = X_contract_cupy_lowk(
                halfrot_cgtoa,
                phikr_kpq,
                MPQ_iq,
                phiki_kpq,
                cgto,
                Ghalfa_batch,
                G_kpq_kprimepq_chunk,
                buff1,
                buff2,
                slices_isdf,
            )
            exx -= exx_iq

        for iq in range(len(Sset), len(Sset) + len(Qplus)):
            iq_real = Qplus[iq - len(Sset)]
            ikpq = kpq_mat[iq_real]
            phikr_kpq = cgto[ikpq]
            phiki_kpq = halfrot_cgtoa[ikpq]
            kpq_idx = kpq_mat[k_idx, iq_real]
            kprimepq_idx = kpq_mat[kprime_idx, iq_real]
            G_kpq_kprimepq_chunk = Ghalfa_batch[w_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
            MPQ_iq = MPQ[iq]
            exx_iq = X_contract_cupy_lowk(
                halfrot_cgtoa,
                phikr_kpq,
                MPQ_iq,
                phiki_kpq,
                cgto,
                Ghalfa_batch,
                G_kpq_kprimepq_chunk,
                buff1,
                buff2,
                slices_isdf,
            )
            exx -= 2.0 * exx_iq

    else:
        w_idx = xp.arange(nwalker)[:, None, None, None, None]
        k_idx = xp.arange(nk)[None, :, None, None, None]
        i_idx = xp.arange(nocc)[None, None, :, None, None]
        kprime_idx = xp.arange(nk)[None, None, None, :, None]
        p_idx = xp.arange(nbsf)[None, None, None, None, :]
        handle = cutensornet.create()

        exx = xp.zeros(nwalker, dtype=numpy.complex128)

        if nk < 64:
            intermediate_mem = nwalker * nisdf * nk * nk * nbsf * 6 * 16 / 1024**3
        else:
            intermediate_mem = nwalker * nisdf * nk * nbsf * 6 * 16 / 1024**3
        free_bytes = xp.cuda.Device().mem_info[0]
        free_gb = free_bytes / 1024**3.0
        max_mem = 0.7 * free_gb
        num_chunks = max(1, ceil(intermediate_mem / max_mem))
        chunk_size = ceil(nwalker / num_chunks)
        nw_left = nwalker
        for i_chunk in range(num_chunks):
            if nw_left == 0:
                break
            n_chunk = min(nw_left, chunk_size)
            nw_left -= n_chunk
            w_sls = xp.arange(nwalker)[i_chunk * chunk_size : i_chunk * chunk_size + n_chunk]
            Ga_chunk = Ghalfa_batch[w_sls]
            w_chunk_idx = xp.arange(n_chunk)[:, None, None, None, None]

            for iq in range(len(Sset)):
                iq_real = Sset[iq]
                ikpq = kpq_mat[iq_real]
                phikr_kpq = cgto[ikpq]
                phiki_kpq = halfrot_cgtoa[ikpq]
                kpq_idx = kpq_mat[k_idx, iq_real]
                kprimepq_idx = kpq_mat[kprime_idx, iq_real]
                G_kpq_kprimepq_chunk = Ga_chunk[w_chunk_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
                MPQ_iq = MPQ[iq]
                network_opts = NetworkOptions(
                    handle=handle, memory_limit=0.7 * xp.cuda.Device().mem_info[0]
                )
                exx[w_sls] -= contract(
                    "kPi, kPp, PQ, KQj, KQq, wkiKq, wKjkp -> w",
                    halfrot_cgtoa.conj(),
                    phikr_kpq,
                    MPQ_iq,
                    phiki_kpq.conj(),
                    cgto,
                    Ga_chunk,
                    G_kpq_kprimepq_chunk,
                    options=network_opts,
                )
                xp.cuda.get_current_stream().synchronize()
                del G_kpq_kprimepq_chunk

            for iq in range(len(Sset), len(Sset) + len(Qplus)):
                iq_real = Qplus[iq - len(Sset)]
                ikpq = kpq_mat[iq_real]
                phikr_kpq = cgto[ikpq]
                phiki_kpq = halfrot_cgtoa[ikpq]
                kpq_idx = kpq_mat[k_idx, iq_real]
                kprimepq_idx = kpq_mat[kprime_idx, iq_real]
                G_kpq_kprimepq_chunk = Ga_chunk[w_chunk_idx, kpq_idx, i_idx, kprimepq_idx, p_idx]
                MPQ_iq = MPQ[iq]
                network_opts = NetworkOptions(
                    handle=handle, memory_limit=0.7 * xp.cuda.Device().mem_info[0]
                )
                exx[w_sls] -= 2.0 * contract(
                    "kPi, kPp, PQ, KQj, KQq, wkiKq, wKjkp -> w",
                    halfrot_cgtoa.conj(),
                    phikr_kpq,
                    MPQ_iq,
                    phiki_kpq.conj(),
                    cgto,
                    Ga_chunk,
                    G_kpq_kprimepq_chunk,
                    options=network_opts,
                )
                xp.cuda.get_current_stream().synchronize()
                del G_kpq_kprimepq_chunk

        cutensornet.destroy(handle)
    return 0.5 * exx / nk


def kpt_isdf_ecoul_kernel_rhf():
    raise NotImplementedError("CPU ISDF Coulomb kernel for RHF not implemented yet.")


@jit(nopython=True, fastmath=True)
def kpt_isdf_ecoul_kernel_uhf():
    raise NotImplementedError("CPU ISDF Coulomb kernel for UHF not implemented yet.")


def local_energy_kpt_single_det_uhf_isdf_gpu(system, hamiltonian, walkers, trial):
    if not config.get_option("use_gpu"):
        raise NotImplementedError("CPU ISDF Coulomb kernel for UHF not implemented yet.")

    nwalkers = walkers.Ghalfa.shape[0]
    nk = hamiltonian.nk
    nalpha = trial.nalpha
    nbeta = trial.nbeta
    nbasis = hamiltonian.nbasis

    if walkers.rhf:
        ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        diagGhalfa = xp.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
        for ik in range(nk):
            diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
        diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
        e1b = 2.0 * diagGhalfa.dot(trial._rH1a.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ecoul = kpt_isdf_ecoul_rhf_kernel_gpu(
            hamiltonian.MPQ,
            trial._rcgtoa,
            hamiltonian.cgto,
            ghalfa,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        exxa = 2.0 * kpt_isdf_exx_kernel_gpu(
            hamiltonian.MPQ,
            trial._rcgtoa,
            hamiltonian.cgto,
            ghalfa,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        e2b = ecoul + exxa
    else:
        ghalfa = walkers.Ghalfa.reshape(nwalkers, nk, nalpha, nk, nbasis)
        ghalfb = walkers.Ghalfb.reshape(nwalkers, nk, nbeta, nk, nbasis)

        diagGhalfa = xp.zeros((nwalkers, nk, nalpha, nbasis), dtype=numpy.complex128)
        diagGhalfb = xp.zeros((nwalkers, nk, nbeta, nbasis), dtype=numpy.complex128)
        for ik in range(nk):
            diagGhalfa[:, ik, :, :] = ghalfa[:, ik, :, ik, :]
            diagGhalfb[:, ik, :, :] = ghalfb[:, ik, :, ik, :]
        diagGhalfa = diagGhalfa.reshape(nwalkers, nk * nalpha * nbasis)
        diagGhalfb = diagGhalfb.reshape(nwalkers, nk * nbeta * nbasis)
        e1b = diagGhalfa.dot(trial._rH1a.ravel())
        e1b += diagGhalfb.dot(trial._rH1b.ravel())
        e1b /= nk
        e1b += hamiltonian.ecore

        ecoul = kpt_isdf_ecoul_kernel_gpu(
            hamiltonian.MPQ,
            trial._rcgtoa,
            trial._rcgtob,
            hamiltonian.cgto,
            ghalfa,
            ghalfb,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        exxa = kpt_isdf_exx_kernel_gpu(
            hamiltonian.MPQ,
            trial._rcgtoa,
            hamiltonian.cgto,
            ghalfa,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )
        exxb = kpt_isdf_exx_kernel_gpu(
            hamiltonian.MPQ,
            trial._rcgtob,
            hamiltonian.cgto,
            ghalfb,
            hamiltonian.ikpq_mat,
            hamiltonian.Sset,
            hamiltonian.Qplus,
        )

        e2b = ecoul + exxa + exxb

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b

    xp._default_memory_pool.free_all_blocks()
    return energy
