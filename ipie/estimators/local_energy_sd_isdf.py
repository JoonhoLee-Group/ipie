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

from math import ceil, sqrt

import numpy

from ipie.utils.backend import arraylib as xp
from ipie.utils.cuquantum_backend import (
    NetworkOptions_optional as NetworkOptions,
    contract_optional as contract,
    cutensornet_optional as cutensornet,
)


def ecoul_kernel_batch_real_isdf_uhf_gpu(
    MPQ, halfrot_cgtoa, halfrot_cgtob, cgto, Ghalfa_batch, Ghalfb_batch
):
    nwalkers = Ghalfa_batch.shape[0]
    ecoul = xp.zeros(nwalkers, dtype=numpy.complex128)
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)

    v_wP_real = contract(
        "Pi, Pp, wip -> wP", halfrot_cgtoa, cgto, Ghalfa_batch.real, options=network_opts
    ) + contract("Pi, Pp, wip -> wP", halfrot_cgtob, cgto, Ghalfb_batch.real, options=network_opts)
    v_wP_imag = contract(
        "Pi, Pp, wip -> wP", halfrot_cgtoa, cgto, Ghalfa_batch.imag, options=network_opts
    ) + contract("Pi, Pp, wip -> wP", halfrot_cgtob, cgto, Ghalfb_batch.imag, options=network_opts)
    v_wP = xp.zeros_like(v_wP_real, dtype=xp.complex128)
    v_wP.real = v_wP_real
    v_wP.imag = v_wP_imag
    ecoul += xp.sum((v_wP @ MPQ) * v_wP, axis=1)
    cutensornet.destroy(handle)
    return 0.5 * ecoul


def ecoul_kernel_batch_real_isdf_rhf_gpu(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch):
    nwalkers = Ghalfa_batch.shape[0]
    ecoul = xp.zeros(nwalkers, dtype=numpy.complex128)
    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)

    v_wP_real = 2.0 * contract(
        "Pi, Pp, wip -> wP", halfrot_cgtoa, cgto, Ghalfa_batch.real, options=network_opts
    )
    v_wP_imag = 2.0 * contract(
        "Pi, Pp, wip -> wP", halfrot_cgtoa, cgto, Ghalfa_batch.imag, options=network_opts
    )
    v_wP = xp.zeros_like(v_wP_real, dtype=xp.complex128)
    v_wP.real = v_wP_real
    v_wP.imag = v_wP_imag
    ecoul += xp.sum((v_wP @ MPQ) * v_wP, axis=1)
    cutensornet.destroy(handle)
    return 0.5 * ecoul


def exx_kernel_batch_real_isdf_uhf_gpu(MPQ, halfrot_cgtoa, cgto, Ghalfa_batch):
    nwalkers = Ghalfa_batch.shape[0]
    exx = xp.zeros(nwalkers, dtype=numpy.complex128)
    nisdf = MPQ.shape[0]
    intermediate_mem = nwalkers * nisdf * nisdf * 16 * 2
    mem_limit = 0.3 * xp.cuda.Device().mem_info[0]
    num_chunks = max(1, ceil(sqrt(intermediate_mem / mem_limit)))
    chunk_size = ceil(nisdf / num_chunks)
    nisdfx_left = nisdf
    slices_x = []
    for i_chunk in range(num_chunks):
        if nisdfx_left == 0:
            break
        nx_chunk = min(nisdfx_left, chunk_size)
        nisdfx_left -= nx_chunk
        slices_x.append(slice(i_chunk * chunk_size, i_chunk * chunk_size + nx_chunk))

    for slicex in slices_x:
        cgto_chunkx = cgto[slicex]
        halfrot_cgto_chunkx = halfrot_cgtoa[slicex]
        for slicey in slices_x:
            cgto_chunky = cgto[slicey]
            halfrot_cgto_chunky = halfrot_cgtoa[slicey]
            MPQ_chunk = MPQ[slicey, slicex].copy()
            MPQ_chunk = MPQ_chunk.ravel()
            Gpsi_wiQ = Ghalfa_batch @ cgto_chunkx.T
            TPQ_chunk = halfrot_cgto_chunky @ Gpsi_wiQ
            TPQ_chunk = TPQ_chunk.copy()
            Gpsi_wjP = Ghalfa_batch @ cgto_chunky.T
            TQP_chunk = halfrot_cgto_chunkx @ Gpsi_wjP
            TQP_chunk = TQP_chunk.copy()
            TQP_chunk = TQP_chunk.transpose(0, 2, 1).copy()
            len_slicex = len(range(*slicex.indices(nisdf)))
            len_slicey = len(range(*slicey.indices(nisdf)))
            TPQ_chunk = TPQ_chunk.reshape(nwalkers, len_slicex * len_slicey)
            TQP_chunk = TQP_chunk.reshape(nwalkers, len_slicex * len_slicey)
            contrib = xp.sum(TPQ_chunk * TQP_chunk * MPQ_chunk[xp.newaxis, :], axis=1)
            exx += contrib
    return 0.5 * exx


def local_energy_single_det_isdf_batch_gpu(system, hamiltonian, walkers, trial, max_mem=2.0):
    nwalkers = walkers.Ghalfa.shape[0]
    if walkers.rhf:
        Ghalfa_batch = walkers.Ghalfa.reshape((nwalkers, -1))

        e1b = 2.0 * Ghalfa_batch.dot(trial._rH1a.ravel())
        e1b += hamiltonian.ecore
    else:
        Ghalfa_batch = walkers.Ghalfa.reshape((nwalkers, -1))
        Ghalfb_batch = walkers.Ghalfb.reshape((nwalkers, -1))

        e1b = Ghalfa_batch.dot(trial._rH1a.ravel())
        e1b += Ghalfb_batch.dot(trial._rH1b.ravel())
        e1b += hamiltonian.ecore

    if walkers.rhf:
        ecoul = ecoul_kernel_batch_real_isdf_rhf_gpu(
            hamiltonian.MPQ, trial._rcgtoa, hamiltonian.cgto, walkers.Ghalfa
        )
        exx = 2.0 * exx_kernel_batch_real_isdf_uhf_gpu(
            hamiltonian.MPQ, trial._rcgtoa, hamiltonian.cgto, walkers.Ghalfa
        )
    else:
        ecoul = ecoul_kernel_batch_real_isdf_uhf_gpu(
            hamiltonian.MPQ,
            trial._rcgtoa,
            trial._rcgtob,
            hamiltonian.cgto,
            walkers.Ghalfa,
            walkers.Ghalfb,
        )
        exx = exx_kernel_batch_real_isdf_uhf_gpu(
            hamiltonian.MPQ,
            trial._rcgtoa,
            hamiltonian.cgto,
            walkers.Ghalfa,
        ) + exx_kernel_batch_real_isdf_uhf_gpu(
            hamiltonian.MPQ,
            trial._rcgtob,
            hamiltonian.cgto,
            walkers.Ghalfb,
        )
    e2b = ecoul - exx

    energy = xp.zeros((nwalkers, 3), dtype=numpy.complex128)
    energy[:, 0] = e1b + e2b
    energy[:, 1] = e1b
    energy[:, 2] = e2b
    return energy
