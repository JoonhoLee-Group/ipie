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
#
# Authors: Joonho Lee <linusjoonho@gmail.com>
#          Fionn Malone <fmalone@google.com>
#          Jinghong Zhang <jinghongzhang@fas.harvard.edu>
#

import numpy
import math
import plum

from numba import jit
from ipie.utils.backend import arraylib as xp
from ipie.utils.backend import synchronize
from ipie.utils.cuquantum_backend import (
    NetworkOptions_optional as NetworkOptions,
    cutensornet_optional as cutensornet,
)
from ipie.utils.contract_gf_cgto import slice_gf_kpq_k_qlis, slice_cgto_kpq
from math import ceil

from ipie.config import config
from ipie.hamiltonians.generic import GenericComplexChol, GenericRealChol, GenericRealISDF
from ipie.walkers.uhf_walkers import UHFWalkers
from ipie.walkers.ghf_walkers import GHFWalkers


def construct_force_bias_batch(hamiltonian, walkers, trial, mpi_handler=None):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.
    trial : class
        Trial wavefunction object.
    mpi_handler : class
        MPIHandler instance.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """

    if walkers.name == "SingleDetWalkerBatch" and trial.name == "MultiSlater":
        if hamiltonian.chunked:
            return construct_force_bias_batch_single_det_chunked(
                hamiltonian, walkers, trial, mpi_handler
            )
        else:
            return construct_force_bias_batch_single_det(hamiltonian, walkers, trial)
    elif walkers.name == "MultiDetTrialWalkerBatch" and trial.name == "MultiSlater":
        return construct_force_bias_batch_multi_det_trial(hamiltonian, walkers, trial)


def construct_force_bias_batch_multi_det_trial(hamiltonian, walkers, trial):
    Ga = walkers.Ga.reshape(walkers.nwalkers, hamiltonian.nbasis**2)
    Gb = walkers.Gb.reshape(walkers.nwalkers, hamiltonian.nbasis**2)
    # Cholesky vectors. [M^2, nchol]
    # Why are there so many transposes here?
    if numpy.isrealobj(hamiltonian.chol):
        vbias_batch = xp.empty((hamiltonian.nchol, walkers.nwalkers), dtype=numpy.complex128)
        vbias_batch.real = hamiltonian.chol.T.dot(Ga.T.real + Gb.T.real)
        vbias_batch.imag = hamiltonian.chol.T.dot(Ga.T.imag + Gb.T.imag)
        vbias_batch = vbias_batch.T.copy()
        return vbias_batch
    else:
        vbias_batch_tmp = hamiltonian.chol.T.dot(Ga.T + Gb.T)
        vbias_batch_tmp = vbias_batch_tmp.T.copy()
        return vbias_batch_tmp


@plum.dispatch
def construct_force_bias_batch_single_det(
    hamiltonian: GenericRealChol, walkers: UHFWalkers, rchola, rcholb
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.
    rchola, rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky for each spin.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    if walkers.rhf:
        Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
        vbias_batch_real = 2.0 * rchola.dot(Ghalfa.T.real)
        vbias_batch_imag = 2.0 * rchola.dot(Ghalfa.T.imag)
        vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
        vbias_batch.real = vbias_batch_real.T.copy()
        vbias_batch.imag = vbias_batch_imag.T.copy()
        synchronize()

        return vbias_batch

    else:
        Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
        Ghalfb = walkers.Ghalfb.reshape(walkers.nwalkers, walkers.ndown * hamiltonian.nbasis)
        vbias_batch_real = rchola.dot(Ghalfa.T.real) + rcholb.dot(Ghalfb.T.real)
        vbias_batch_imag = rchola.dot(Ghalfa.T.imag) + rcholb.dot(Ghalfb.T.imag)
        vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
        vbias_batch.real = vbias_batch_real.T.copy()
        vbias_batch.imag = vbias_batch_imag.T.copy()
        synchronize()
        return vbias_batch


@plum.dispatch
def construct_force_bias_batch_single_det(
    hamiltonian: GenericComplexChol, walkers: UHFWalkers, rAa, rAb, rBa, rBb
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
    Ghalfb = walkers.Ghalfb.reshape(walkers.nwalkers, walkers.ndown * hamiltonian.nbasis)
    vbias_batch = xp.zeros((hamiltonian.nfields, walkers.nwalkers), dtype=Ghalfa.dtype)
    vbias_batch[: hamiltonian.nchol, :] = rAa.dot(Ghalfa.T) + rAb.dot(Ghalfb.T)
    vbias_batch[hamiltonian.nchol :, :] = rBa.dot(Ghalfa.T) + rBb.dot(Ghalfb.T)
    vbias_batch = vbias_batch.T.copy()
    synchronize()
    return vbias_batch


@plum.dispatch
def construct_force_bias_batch_single_det(
    hamiltonian: GenericRealISDF,
    walkers: UHFWalkers,
    rcgtoa,
    rcgtob,
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.
    rchola, rcholb : :class:`numpy.ndarray`
        Half-rotated cholesky for each spin.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    if walkers.rhf:
        Ghalfa = walkers.Ghalfa
        handle = cutensornet.create()
        network_opts = NetworkOptions(handle=handle)
        vbias_batch_real = 2.0 * cutensornet.contract(
            "Pi, Pr, Pg, wir -> wg",
            rcgtoa,
            hamiltonian.cgto,
            hamiltonian.cholM,
            Ghalfa.real,
            options=network_opts,
        )
        vbias_batch_imag = 2.0 * cutensornet.contract(
            "Pi, Pr, Pg, wir -> wg",
            rcgtoa,
            hamiltonian.cgto,
            hamiltonian.cholM,
            Ghalfa.imag,
            options=network_opts,
        )
        vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
        vbias_batch.real = vbias_batch_real
        vbias_batch.imag = vbias_batch_imag
        cutensornet.destroy(handle)
        synchronize()

        return vbias_batch

    else:
        Ghalfa = walkers.Ghalfa
        Ghalfb = walkers.Ghalfb
        handle = cutensornet.create()
        network_opts = NetworkOptions(handle=handle)
        vbias_batch_real = cutensornet.contract(
            "Pi, Pr, Pg, wir -> wg",
            rcgtoa,
            hamiltonian.cgto,
            hamiltonian.cholM,
            Ghalfa.real,
            options=network_opts,
        ) + cutensornet.contract(
            "Pi, Pr, Pg, wir -> wg",
            rcgtob,
            hamiltonian.cgto,
            hamiltonian.cholM,
            Ghalfb.real,
            options=network_opts,
        )
        vbias_batch_imag = cutensornet.contract(
            "Pi, Pr, Pg, wir -> wg",
            rcgtoa,
            hamiltonian.cgto,
            hamiltonian.cholM,
            Ghalfa.imag,
            options=network_opts,
        ) + cutensornet.contract(
            "Pi, Pr, Pg, wir -> wg",
            rcgtob,
            hamiltonian.cgto,
            hamiltonian.cholM,
            Ghalfb.imag,
            options=network_opts,
        )
        vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
        vbias_batch.real = vbias_batch_real
        vbias_batch.imag = vbias_batch_imag
        cutensornet.destroy(handle)
        synchronize()
        return vbias_batch


@plum.dispatch
def construct_force_bias_batch_single_det(hamiltonian: GenericRealChol, walkers: GHFWalkers):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    Ga = walkers.Ga
    Gb = walkers.Gb
    Gcharge = (Ga + Gb).reshape(walkers.nwalkers, -1)  # (nwalkers, nbasis**2)

    vbias_batch = numpy.zeros((walkers.nwalkers, hamiltonian.nfields), dtype=Ga.dtype)
    vbias_real = xp.einsum("pl, wp->wl", hamiltonian.chol, Gcharge.real)
    vbias_imag = xp.einsum("pl, wp->wl", hamiltonian.chol, Gcharge.imag)
    vbias_batch.real = vbias_real
    vbias_batch.imag = vbias_imag
    synchronize()
    return vbias_batch


@plum.dispatch
def construct_force_bias_batch_single_det(hamiltonian: GenericComplexChol, walkers: GHFWalkers):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    Ga = walkers.Ga
    Gb = walkers.Gb
    Gcharge = (Ga + Gb).reshape(walkers.nwalkers, -1)  # (nwalkers, nbasis**2)

    vbias_batch = numpy.zeros((walkers.nwalkers, hamiltonian.nfields), dtype=Ga.dtype)
    vbias_A = xp.einsum("pl, wp->wl", hamiltonian.A, Gcharge)
    vbias_B = xp.einsum("pl, wp->wl", hamiltonian.B, Gcharge)
    vbias_batch[:, : hamiltonian.nchol] = vbias_A
    vbias_batch[:, hamiltonian.nchol :] = vbias_B
    synchronize()
    return vbias_batch


def construct_force_bias_kpt_batch_single_det(
    hamiltonian: "KptComplexChol", walkers: "UHFWalkers", trial: "KptSingleDet"
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    vbias_plus : :class:`numpy.ndarray`
        Force bias for Lplus.
    vbias_minus : :class:`numpy.ndarray`
        Force bias for Lminus.
    """
    if walkers.rhf:
        vbias = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.nk), dtype=numpy.complex128
        )
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalf_reshape = walkers.Ghalfa.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
        )
        for iq in range(hamiltonian.nk):
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[ik, iq]
                vbias[:, :, iq] += 2.0 * xp.einsum(
                    "gip, aip -> ga",
                    trial._rchola[:, ik, :, iq, :],
                    Ghalf_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                )
        synchronize()
        imq = hamiltonian.imq_vec
        vbias_plus = 0.5 * 1j * (vbias + vbias[:, :, imq])
        vbias_minus = 0.5 * (vbias - vbias[:, :, imq])
        return vbias_plus, vbias_minus

    else:
        vbias = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.nk), dtype=numpy.complex128
        )
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalfa_reshape = walkers.Ghalfa.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
        )
        Ghalfb_reshape = walkers.Ghalfb.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nbeta, hamiltonian.nk, hamiltonian.nbasis
        )
        for iq in range(hamiltonian.nk):
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[ik, iq]
                vbias[:, :, iq] += xp.einsum(
                    "gip, aip -> ag",
                    trial._rchola[:, ik, :, iq, :],
                    Ghalfa_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                ) + xp.einsum(
                    "gip, bip -> bg",
                    trial._rcholb[:, ik, :, iq, :],
                    Ghalfb_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                )
        synchronize()
        imq = hamiltonian.imq_vec
        vbias_plus = 0.5 * 1j * (vbias + vbias[:, :, imq])
        vbias_minus = 0.5 * (vbias - vbias[:, :, imq])
        return vbias_plus, vbias_minus


def construct_force_bias_kptsymm_batch_single_det(
    hamiltonian: "KptComplexCholSymm", walkers: "UHFWalkers", trial: "KptSingleDet"
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    vbias_plus : :class:`numpy.ndarray`
        Force bias for Lplus.
    vbias_minus : :class:`numpy.ndarray`
        Force bias for Lminus.
    """
    if walkers.rhf:
        vbias_plus = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )
        vbias_minus = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalf_reshape = walkers.Ghalfa.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
        )
        for iq in range(len(hamiltonian.Sset)):
            iq_real = hamiltonian.Sset[iq]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                vbias_plus[:, :, iq] += 1.0j * xp.einsum(
                    "igp, aip -> ag",
                    trial._rchola[iq, ik],
                    Ghalfa_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                )
                vbias_plus[:, :, iq] += 1.0j * xp.einsum(
                    "pgi, aip -> ag",
                    trial._rcholbara[iq, ik],
                    Ghalfa_reshape[:, ikpq, :, ik, :],
                    optimize=True,
                )

                vbias_minus[:, :, iq] += xp.einsum(
                    "igp, aip -> ag",
                    trial._rchola[iq, ik],
                    Ghalfa_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                )
                vbias_minus[:, :, iq] -= xp.einsum(
                    "pgi, aip -> ag",
                    trial._rcholbara[iq, ik],
                    Ghalfa_reshape[:, ikpq, :, ik, :],
                    optimize=True,
                )

        for iq in range(len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)):
            iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                vbias_plus[:, :, iq] += (
                    1.0j
                    * xp.sqrt(2)
                    * xp.einsum(
                        "igp, aip -> ag",
                        trial._rchola[iq, ik],
                        Ghalfa_reshape[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                )
                vbias_plus[:, :, iq] += (
                    1.0j
                    * xp.sqrt(2)
                    * xp.einsum(
                        "pgi, aip -> ag",
                        trial._rcholbara[iq, ik],
                        Ghalfa_reshape[:, ikpq, :, ik, :],
                        optimize=True,
                    )
                )

                vbias_minus[:, :, iq] += xp.sqrt(2) * xp.einsum(
                    "igp, aip -> ag",
                    trial._rchola[iq, ik],
                    Ghalfa_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                )
                vbias_minus[:, :, iq] -= xp.sqrt(2) * xp.einsum(
                    "pgi, aip -> ag",
                    trial._rcholbara[iq, ik],
                    Ghalfa_reshape[:, ikpq, :, ik, :],
                    optimize=True,
                )
        synchronize()
        return vbias_plus, vbias_minus

    else:
        vbias_plus = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )
        vbias_minus = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalfa_reshape = walkers.Ghalfa.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
        )
        Ghalfb_reshape = walkers.Ghalfb.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nbeta, hamiltonian.nk, hamiltonian.nbasis
        )
        for iq in range(len(hamiltonian.Sset)):
            iq_real = hamiltonian.Sset[iq]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                vbias_plus[:, :, iq] += 0.5j * (
                    xp.einsum(
                        "igp, aip -> ag",
                        trial._rchola[iq, ik],
                        Ghalfa_reshape[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                    + xp.einsum(
                        "igp, bip -> bg",
                        trial._rcholb[iq, ik],
                        Ghalfb_reshape[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                )
                vbias_plus[:, :, iq] += 0.5j * (
                    xp.einsum(
                        "pgi, aip -> ag",
                        trial._rcholbara[iq, ik],
                        Ghalfa_reshape[:, ikpq, :, ik, :],
                        optimize=True,
                    )
                    + xp.einsum(
                        "pgi, bip -> bg",
                        trial._rcholbarb[iq, ik],
                        Ghalfb_reshape[:, ikpq, :, ik, :],
                        optimize=True,
                    )
                )

                vbias_minus[:, :, iq] += 0.5 * (
                    xp.einsum(
                        "igp, aip -> ag",
                        trial._rchola[iq, ik],
                        Ghalfa_reshape[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                    + xp.einsum(
                        "igp, bip -> bg",
                        trial._rcholb[iq, ik],
                        Ghalfb_reshape[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                )
                vbias_minus[:, :, iq] -= 0.5 * (
                    xp.einsum(
                        "pgi, aip -> ag",
                        trial._rcholbara[iq, ik],
                        Ghalfa_reshape[:, ikpq, :, ik, :],
                        optimize=True,
                    )
                    + xp.einsum(
                        "pgi, bip -> bg",
                        trial._rcholbarb[iq, ik],
                        Ghalfb_reshape[:, ikpq, :, ik, :],
                        optimize=True,
                    )
                )

        for iq in range(len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)):
            iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                vbias_plus[:, :, iq] += (
                    0.5j
                    * xp.sqrt(2)
                    * (
                        xp.einsum(
                            "igp, aip -> ag",
                            trial._rchola[iq, ik],
                            Ghalfa_reshape[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                        + xp.einsum(
                            "igp, bip -> bg",
                            trial._rcholb[iq, ik],
                            Ghalfb_reshape[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                    )
                )
                vbias_plus[:, :, iq] += (
                    0.5j
                    * xp.sqrt(2)
                    * (
                        xp.einsum(
                            "pgi, aip -> ag",
                            trial._rcholbara[iq, ik],
                            Ghalfa_reshape[:, ikpq, :, ik, :],
                            optimize=True,
                        )
                        + xp.einsum(
                            "pgi, bip -> bg",
                            trial._rcholbarb[iq, ik],
                            Ghalfb_reshape[:, ikpq, :, ik, :],
                            optimize=True,
                        )
                    )
                )

                vbias_minus[:, :, iq] += (
                    0.5
                    * xp.sqrt(2)
                    * (
                        xp.einsum(
                            "igp, aip -> ag",
                            trial._rchola[iq, ik],
                            Ghalfa_reshape[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                        + xp.einsum(
                            "igp, bip -> bg",
                            trial._rcholb[iq, ik],
                            Ghalfb_reshape[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                    )
                )
                vbias_minus[:, :, iq] -= (
                    0.5
                    * xp.sqrt(2)
                    * (
                        xp.einsum(
                            "pgi, aip -> ag",
                            trial._rcholbara[iq, ik],
                            Ghalfa_reshape[:, ikpq, :, ik, :],
                            optimize=True,
                        )
                        + xp.einsum(
                            "pgi, bip -> bg",
                            trial._rcholbarb[iq, ik],
                            Ghalfb_reshape[:, ikpq, :, ik, :],
                            optimize=True,
                        )
                    )
                )
        synchronize()
        return vbias_plus, vbias_minus


def construct_force_bias_kptisdf_batch_single_det(
    hamiltonian: "KptISDF", walkers: "UHFWalkers", trial: "KptSingleDet", max_mem=4.0
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    vbias_plus : :class:`numpy.ndarray`
        Force bias for Lplus.
    vbias_minus : :class:`numpy.ndarray`
        Force bias for Lminus.
    """
    if walkers.rhf:
        if config.get_option("use_gpu"):
            nwalkers = walkers.nwalkers
            vbias_plus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            vbias_minus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            # ghalf shape: nwalkers, nk nup, hamiltonian.nk, nbsf
            Ghalfa_reshape = walkers.Ghalfa.reshape(
                nwalkers,
                hamiltonian.nk,
                trial.nalpha,
                hamiltonian.nk,
                hamiltonian.halfrot_cgto.shape[-1],
            )

            # slice Sset and Qplus according to memory
            mem_cost_Sset = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Sset)
                * hamiltonian.nk
                * (trial.nalpha)
                * 2
                * 16
                / (1024**3)
            )
            mem_cost_Qplus = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Qplus)
                * hamiltonian.nk
                * (trial.nalpha)
                * 2
                * 16
                / (1024**3)
            )

            num_nq_chunks_Sset = max(1, ceil(mem_cost_Sset / max_mem))
            nq_chunk_Sset_size = ceil(len(hamiltonian.Sset) / num_nq_chunks_Sset)
            nq_left = len(hamiltonian.Sset)
            handle = cutensornet.create()
            network_opts = NetworkOptions(handle=handle)
            if len(hamiltonian.Sset) > 0:
                for i in range(num_nq_chunks_Sset):
                    nq_chunk = min(nq_left, nq_chunk_Sset_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Sset[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    kpq_slice = hamiltonian.ikpq_mat[q_sls]
                    ga_kmq = slice_gf_kpq_k_qlis(
                        Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat
                    )  # q, k, w, p, r
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    X_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kmq,
                        options=network_opts,
                    )
                    ga_kpq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikpq_mat)
                    rcgtoa_kpq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikpq_mat, q_sls)
                    Y_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kpq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kpq,
                        options=network_opts,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    vbias_plus[
                        :, :, i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ] += 2.0j * cutensornet.contract(
                        "qwP, qPg -> wgq", X_wPa, L_q, options=network_opts
                    )

            num_nq_chunks_Qplus = max(1, ceil(mem_cost_Qplus / max_mem))
            nq_chunk_Qplus_size = ceil(len(hamiltonian.Qplus) / num_nq_chunks_Qplus)
            nq_left = len(hamiltonian.Qplus)
            if len(hamiltonian.Qplus) > 0:
                for i in range(num_nq_chunks_Qplus):
                    nq_chunk = min(nq_left, nq_chunk_Qplus_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Qplus[
                        i * nq_chunk_Qplus_size : i * nq_chunk_Qplus_size + nq_chunk
                    ]
                    kpq_slice = hamiltonian.ikpq_mat[q_sls]
                    ga_kmq = slice_gf_kpq_k_qlis(
                        Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat
                    )  # q, k, w, p, r
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    ga_kpq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikpq_mat)
                    rcgtoa_kpq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikpq_mat, q_sls)
                    X_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kmq,
                        options=network_opts,
                    )
                    Y_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kpq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kpq,
                        options=network_opts,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset)
                    ]
                    v1 = cutensornet.contract("qwP, qPg -> wgq", X_wPa, L_q, options=network_opts)
                    v2 = cutensornet.contract(
                        "qwP, qPg -> wgq", Y_wPa, L_q.conj(), options=network_opts
                    )
                    vbias_plus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        1j * xp.sqrt(2) * (v1 + v2)
                    )
                    vbias_minus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        1.0 * xp.sqrt(2) * (v1 - v2)
                    )
            cutensornet.destroy(handle)
            synchronize()
            return vbias_plus, vbias_minus
        else:
            pass
    else:
        if config.get_option("use_gpu"):
            nwalkers = walkers.nwalkers
            vbias_plus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            vbias_minus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            # ghalf shape: nwalkers, nk nup, hamiltonian.nk, nbsf
            Ghalfa_reshape = walkers.Ghalfa.reshape(
                nwalkers,
                hamiltonian.nk,
                trial.nalpha,
                hamiltonian.nk,
                hamiltonian.halfrot_cgto.shape[-1],
            )
            Ghalfb_reshape = walkers.Ghalfb.reshape(
                nwalkers,
                hamiltonian.nk,
                trial.nbeta,
                hamiltonian.nk,
                hamiltonian.halfrot_cgto.shape[-1],
            )

            # slice Sset and Qplus according to memory
            mem_cost_Sset = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Sset)
                * hamiltonian.nk
                * (trial.nalpha + trial.nbeta)
                * 16
                / (1024**3)
            )
            mem_cost_Qplus = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Qplus)
                * hamiltonian.nk
                * (trial.nalpha + trial.nbeta)
                * 16
                / (1024**3)
            )

            num_nq_chunks_Sset = max(1, ceil(mem_cost_Sset / max_mem))
            nq_chunk_Sset_size = ceil(len(hamiltonian.Sset) / num_nq_chunks_Sset)
            nq_left = len(hamiltonian.Sset)
            handle = cutensornet.create()
            network_opts = NetworkOptions(handle=handle)
            if len(hamiltonian.Sset) > 0:
                for i in range(num_nq_chunks_Sset):
                    nq_chunk = min(nq_left, nq_chunk_Sset_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Sset[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    kpq_slice = hamiltonian.ikpq_mat[q_sls]
                    ga_kmq = slice_gf_kpq_k_qlis(
                        Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat
                    )  # q, k, w, p, r
                    gb_kmq = slice_gf_kpq_k_qlis(Ghalfb_reshape, q_sls, hamiltonian.ikmq_mat)
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    rcgtob_kmq = slice_cgto_kpq(trial._rcgtob, hamiltonian.ikmq_mat, q_sls)
                    X_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kmq,
                        options=network_opts,
                    )
                    X_wPb = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtob_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        gb_kmq,
                        options=network_opts,
                    )
                    ga_kpq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikpq_mat)
                    gb_kpq = slice_gf_kpq_k_qlis(Ghalfb_reshape, q_sls, hamiltonian.ikpq_mat)
                    rcgtoa_kpq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikpq_mat, q_sls)
                    rcgtob_kpq = slice_cgto_kpq(trial._rcgtob, hamiltonian.ikpq_mat, q_sls)
                    Y_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kpq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kpq,
                        options=network_opts,
                    )
                    Y_wPb = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtob_kpq.conj(),
                        hamiltonian.halfrot_cgto,
                        gb_kpq,
                        options=network_opts,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    vbias_plus[
                        :, :, i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ] += 1j * cutensornet.contract(
                        "qwP, qPg -> wgq", X_wPa + X_wPb, L_q, options=network_opts
                    )

            num_nq_chunks_Qplus = max(1, ceil(mem_cost_Qplus / max_mem))
            nq_chunk_Qplus_size = ceil(len(hamiltonian.Qplus) / num_nq_chunks_Qplus)
            nq_left = len(hamiltonian.Qplus)
            if len(hamiltonian.Qplus) > 0:
                for i in range(num_nq_chunks_Qplus):
                    nq_chunk = min(nq_left, nq_chunk_Qplus_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Qplus[
                        i * nq_chunk_Qplus_size : i * nq_chunk_Qplus_size + nq_chunk
                    ]
                    kpq_slice = hamiltonian.ikpq_mat[q_sls]
                    ga_kmq = slice_gf_kpq_k_qlis(
                        Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat
                    )  # q, k, w, p, r
                    gb_kmq = slice_gf_kpq_k_qlis(Ghalfb_reshape, q_sls, hamiltonian.ikmq_mat)
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    rcgtob_kmq = slice_cgto_kpq(trial._rcgtob, hamiltonian.ikmq_mat, q_sls)
                    ga_kpq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikpq_mat)
                    gb_kpq = slice_gf_kpq_k_qlis(Ghalfb_reshape, q_sls, hamiltonian.ikpq_mat)
                    rcgtoa_kpq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikpq_mat, q_sls)
                    rcgtob_kpq = slice_cgto_kpq(trial._rcgtob, hamiltonian.ikpq_mat, q_sls)
                    X_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kmq,
                        options=network_opts,
                    )
                    X_wPb = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtob_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        gb_kmq,
                        options=network_opts,
                    )
                    Y_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kpq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kpq,
                        options=network_opts,
                    )
                    Y_wPb = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtob_kpq.conj(),
                        hamiltonian.halfrot_cgto,
                        gb_kpq,
                        options=network_opts,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset)
                    ]
                    v1 = cutensornet.contract(
                        "qwP, qPg -> wgq", X_wPa + X_wPb, L_q, options=network_opts
                    )
                    v2 = cutensornet.contract(
                        "qwP, qPg -> wgq", Y_wPa + Y_wPb, L_q.conj(), options=network_opts
                    )
                    vbias_plus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        0.5j * xp.sqrt(2) * (v1 + v2)
                    )
                    vbias_minus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        0.5 * xp.sqrt(2) * (v1 - v2)
                    )
            cutensornet.destroy(handle)
            synchronize()
            return vbias_plus, vbias_minus
            # vbias_plus = xp.zeros((walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128)
            # vbias_minus = xp.zeros((walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128)
            # # ghalf shape: nwalkers, nk, nup, nk, nbsf
            # Ghalfa_reshape = walkers.Ghalfa.reshape(walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis)
            # Ghalfb_reshape = walkers.Ghalfb.reshape(walkers.nwalkers, hamiltonian.nk, trial.nbeta, hamiltonian.nk, hamiltonian.nbasis)
            # handle = cutensornet.create()
            # network_opts = NetworkOptions(handle=handle)
            # for iq in range(len(hamiltonian.Sset)):
            #     iq_real = hamiltonian.Sset[iq]
            #     ikpq = hamiltonian.ikpq_mat[iq_real]
            #     ga_kpq = slice_gf_kpq_k_given_q(Ghalfa_reshape, iq_real, hamiltonian.ikpq_mat)
            #     gb_kpq = slice_gf_kpq_k_given_q(Ghalfb_reshape, iq_real, hamiltonian.ikpq_mat)
            #     cgto_kpq = hamiltonian.halfrot_cgto[ikpq]
            #     X_wPa = contract_gf_cgto12_k_kpq(ga_kpq, trial._rcgtoa, cgto_kpq, iq_real, network_opts)
            #     X_wPb = contract_gf_cgto12_k_kpq(gb_kpq, trial._rcgtob, cgto_kpq, iq_real, network_opts)
            #     L_q = hamiltonian.cholM[iq]
            #     vbias_plus[:, :, iq] += 1j * (X_wPa + X_wPb).dot(L_q)
            #     vbias_minus[:, :, iq] += xp.zeros_like(vbias_plus[:, :, iq])

            # for iq in range(len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)):
            #     iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
            #     ikpq = hamiltonian.ikpq_mat[iq_real]
            #     cgto_kpq = hamiltonian.halfrot_cgto[ikpq]
            #     rcgtoa_kpq = trial._rcgtoa[ikpq]
            #     rcgtob_kpq = trial._rcgtob[ikpq]
            #     ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_reshape, iq_real, hamiltonian.ikpq_mat)
            #     Y_wPa = contract_gf_cgto12_kpq_k(ga_kpq_k, rcgtoa_kpq, hamiltonian.halfrot_cgto, iq_real, network_opts)
            #     # del ga_kpq_k
            #     gb_kpq_k = slice_gf_kpq_k_given_q(Ghalfb_reshape, iq_real, hamiltonian.ikpq_mat)
            #     Y_wPb = contract_gf_cgto12_kpq_k(gb_kpq_k, rcgtob_kpq, hamiltonian.halfrot_cgto, iq_real, network_opts)
            #     # del gb_kpq_k
            #     ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_reshape, iq_real, hamiltonian.ikpq_mat)
            #     X_wPa = contract_gf_cgto12_k_kpq(ga_k_kpq, trial._rcgtoa, cgto_kpq, iq_real, network_opts)
            #     # del ga_k_kpq
            #     gb_k_kpq = slice_gf_k_kpq_given_q(Ghalfb_reshape, iq_real, hamiltonian.ikpq_mat)
            #     X_wPb = contract_gf_cgto12_k_kpq(gb_k_kpq, trial._rcgtob, cgto_kpq, iq_real, network_opts)
            #     # del gb_k_kpq
            #     L_q = hamiltonian.cholM[iq]
            #     v1 = (X_wPa + X_wPb).dot(L_q)
            #     v2 = (Y_wPa + Y_wPb).dot(L_q.conj())
            #     vbias_plus[:, :, iq] += .5j * xp.sqrt(2) * (v1 + v2)
            #     vbias_minus[:, :, iq] += .5 * xp.sqrt(2) * (v1 - v2)
            # cutensornet.destroy(handle)
            # synchronize()
            # return vbias_plus, vbias_minus
        else:
            pass


@plum.dispatch
def construct_force_bias_batch_single_det(
    hamiltonian: GenericComplexChol, walkers: UHFWalkers, rAa, rAb, rBa, rBb
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
    Ghalfb = walkers.Ghalfb.reshape(walkers.nwalkers, walkers.ndown * hamiltonian.nbasis)
    vbias_batch = xp.zeros((hamiltonian.nfields, walkers.nwalkers), dtype=Ghalfa.dtype)
    vbias_batch[: hamiltonian.nchol, :] = rAa.dot(Ghalfa.T) + rAb.dot(Ghalfb.T)
    vbias_batch[hamiltonian.nchol :, :] = rBa.dot(Ghalfa.T) + rBb.dot(Ghalfb.T)
    vbias_batch = vbias_batch.T.copy()
    synchronize()
    return vbias_batch


@plum.dispatch
def construct_force_bias_batch_single_det(hamiltonian: GenericRealChol, walkers: GHFWalkers):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    Ga = walkers.Ga
    Gb = walkers.Gb
    Gcharge = (Ga + Gb).reshape(walkers.nwalkers, -1)  # (nwalkers, nbasis**2)

    vbias_batch = numpy.zeros((walkers.nwalkers, hamiltonian.nfields), dtype=Ga.dtype)
    vbias_real = xp.einsum("pl, wp->wl", hamiltonian.chol, Gcharge.real)
    vbias_imag = xp.einsum("pl, wp->wl", hamiltonian.chol, Gcharge.imag)
    vbias_batch.real = vbias_real
    vbias_batch.imag = vbias_imag
    synchronize()
    return vbias_batch


@plum.dispatch
def construct_force_bias_batch_single_det(hamiltonian: GenericComplexChol, walkers: GHFWalkers):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    Ga = walkers.Ga
    Gb = walkers.Gb
    Gcharge = (Ga + Gb).reshape(walkers.nwalkers, -1)  # (nwalkers, nbasis**2)

    vbias_batch = numpy.zeros((walkers.nwalkers, hamiltonian.nfields), dtype=Ga.dtype)
    vbias_A = xp.einsum("pl, wp->wl", hamiltonian.A, Gcharge)
    vbias_B = xp.einsum("pl, wp->wl", hamiltonian.B, Gcharge)
    vbias_batch[:, : hamiltonian.nchol] = vbias_A
    vbias_batch[:, hamiltonian.nchol :] = vbias_B
    synchronize()
    return vbias_batch


def construct_force_bias_kpt_batch_single_det(
    hamiltonian: "KptComplexChol", walkers: "UHFWalkers", trial: "KptSingleDet"
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    vbias_plus : :class:`numpy.ndarray`
        Force bias for Lplus.
    vbias_minus : :class:`numpy.ndarray`
        Force bias for Lminus.
    """
    if walkers.rhf:
        vbias = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.nk), dtype=numpy.complex128
        )
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalf_reshape = walkers.Ghalfa.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
        )
        for iq in range(hamiltonian.nk):
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[ik, iq]
                vbias[:, :, iq] += 2.0 * xp.einsum(
                    "gip, aip -> ga",
                    trial._rchola[:, ik, :, iq, :],
                    Ghalf_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                )
        synchronize()
        imq = hamiltonian.imq_vec
        vbias_plus = 0.5 * 1j * (vbias + vbias[:, :, imq])
        vbias_minus = 0.5 * (vbias - vbias[:, :, imq])
        return vbias_plus, vbias_minus

    else:
        vbias = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.nk), dtype=numpy.complex128
        )
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalfa_reshape = walkers.Ghalfa.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
        )
        Ghalfb_reshape = walkers.Ghalfb.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nbeta, hamiltonian.nk, hamiltonian.nbasis
        )
        for iq in range(hamiltonian.nk):
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[ik, iq]
                vbias[:, :, iq] += xp.einsum(
                    "gip, aip -> ag",
                    trial._rchola[:, ik, :, iq, :],
                    Ghalfa_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                ) + xp.einsum(
                    "gip, bip -> bg",
                    trial._rcholb[:, ik, :, iq, :],
                    Ghalfb_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                )
        synchronize()
        imq = hamiltonian.imq_vec
        vbias_plus = 0.5 * 1j * (vbias + vbias[:, :, imq])
        vbias_minus = 0.5 * (vbias - vbias[:, :, imq])
        return vbias_plus, vbias_minus


def construct_force_bias_kptsymm_batch_single_det(
    hamiltonian: "KptComplexCholSymm", walkers: "UHFWalkers", trial: "KptSingleDet"
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    vbias_plus : :class:`numpy.ndarray`
        Force bias for Lplus.
    vbias_minus : :class:`numpy.ndarray`
        Force bias for Lminus.
    """
    if walkers.rhf:
        vbias_plus = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )
        vbias_minus = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalfa_reshape = walkers.Ghalfa.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
        )
        for iq in range(len(hamiltonian.Sset)):
            iq_real = hamiltonian.Sset[iq]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                vbias_plus[:, :, iq] += 1.0j * xp.einsum(
                    "igp, aip -> ag",
                    trial._rchola[iq, ik],
                    Ghalfa_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                )
                vbias_plus[:, :, iq] += 1.0j * xp.einsum(
                    "pgi, aip -> ag",
                    trial._rcholbara[iq, ik],
                    Ghalfa_reshape[:, ikpq, :, ik, :],
                    optimize=True,
                )

                vbias_minus[:, :, iq] += xp.einsum(
                    "igp, aip -> ag",
                    trial._rchola[iq, ik],
                    Ghalfa_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                )
                vbias_minus[:, :, iq] -= xp.einsum(
                    "pgi, aip -> ag",
                    trial._rcholbara[iq, ik],
                    Ghalfa_reshape[:, ikpq, :, ik, :],
                    optimize=True,
                )

        for iq in range(len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)):
            iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                vbias_plus[:, :, iq] += (
                    1.0j
                    * xp.sqrt(2)
                    * xp.einsum(
                        "igp, aip -> ag",
                        trial._rchola[iq, ik],
                        Ghalfa_reshape[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                )
                vbias_plus[:, :, iq] += (
                    1.0j
                    * xp.sqrt(2)
                    * xp.einsum(
                        "pgi, aip -> ag",
                        trial._rcholbara[iq, ik],
                        Ghalfa_reshape[:, ikpq, :, ik, :],
                        optimize=True,
                    )
                )

                vbias_minus[:, :, iq] += xp.sqrt(2) * xp.einsum(
                    "igp, aip -> ag",
                    trial._rchola[iq, ik],
                    Ghalfa_reshape[:, ik, :, ikpq, :],
                    optimize=True,
                )
                vbias_minus[:, :, iq] -= xp.sqrt(2) * xp.einsum(
                    "pgi, aip -> ag",
                    trial._rcholbara[iq, ik],
                    Ghalfa_reshape[:, ikpq, :, ik, :],
                    optimize=True,
                )
        synchronize()
        return vbias_plus, vbias_minus

    else:
        vbias_plus = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )
        vbias_minus = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )
        # ghalf shape: nwalkers, nk, nup, nk, nbsf
        Ghalfa_reshape = walkers.Ghalfa.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
        )
        Ghalfb_reshape = walkers.Ghalfb.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nbeta, hamiltonian.nk, hamiltonian.nbasis
        )
        for iq in range(len(hamiltonian.Sset)):
            iq_real = hamiltonian.Sset[iq]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                vbias_plus[:, :, iq] += 0.5j * (
                    xp.einsum(
                        "igp, aip -> ag",
                        trial._rchola[iq, ik],
                        Ghalfa_reshape[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                    + xp.einsum(
                        "igp, bip -> bg",
                        trial._rcholb[iq, ik],
                        Ghalfb_reshape[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                )
                vbias_plus[:, :, iq] += 0.5j * (
                    xp.einsum(
                        "pgi, aip -> ag",
                        trial._rcholbara[iq, ik],
                        Ghalfa_reshape[:, ikpq, :, ik, :],
                        optimize=True,
                    )
                    + xp.einsum(
                        "pgi, bip -> bg",
                        trial._rcholbarb[iq, ik],
                        Ghalfb_reshape[:, ikpq, :, ik, :],
                        optimize=True,
                    )
                )

                vbias_minus[:, :, iq] += 0.5 * (
                    xp.einsum(
                        "igp, aip -> ag",
                        trial._rchola[iq, ik],
                        Ghalfa_reshape[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                    + xp.einsum(
                        "igp, bip -> bg",
                        trial._rcholb[iq, ik],
                        Ghalfb_reshape[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                )
                vbias_minus[:, :, iq] -= 0.5 * (
                    xp.einsum(
                        "pgi, aip -> ag",
                        trial._rcholbara[iq, ik],
                        Ghalfa_reshape[:, ikpq, :, ik, :],
                        optimize=True,
                    )
                    + xp.einsum(
                        "pgi, bip -> bg",
                        trial._rcholbarb[iq, ik],
                        Ghalfb_reshape[:, ikpq, :, ik, :],
                        optimize=True,
                    )
                )

        for iq in range(len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)):
            iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
            for ik in range(hamiltonian.nk):
                ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                vbias_plus[:, :, iq] += (
                    0.5j
                    * xp.sqrt(2)
                    * (
                        xp.einsum(
                            "igp, aip -> ag",
                            trial._rchola[iq, ik],
                            Ghalfa_reshape[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                        + xp.einsum(
                            "igp, bip -> bg",
                            trial._rcholb[iq, ik],
                            Ghalfb_reshape[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                    )
                )
                vbias_plus[:, :, iq] += (
                    0.5j
                    * xp.sqrt(2)
                    * (
                        xp.einsum(
                            "pgi, aip -> ag",
                            trial._rcholbara[iq, ik],
                            Ghalfa_reshape[:, ikpq, :, ik, :],
                            optimize=True,
                        )
                        + xp.einsum(
                            "pgi, bip -> bg",
                            trial._rcholbarb[iq, ik],
                            Ghalfb_reshape[:, ikpq, :, ik, :],
                            optimize=True,
                        )
                    )
                )

                vbias_minus[:, :, iq] += (
                    0.5
                    * xp.sqrt(2)
                    * (
                        xp.einsum(
                            "igp, aip -> ag",
                            trial._rchola[iq, ik],
                            Ghalfa_reshape[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                        + xp.einsum(
                            "igp, bip -> bg",
                            trial._rcholb[iq, ik],
                            Ghalfb_reshape[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                    )
                )
                vbias_minus[:, :, iq] -= (
                    0.5
                    * xp.sqrt(2)
                    * (
                        xp.einsum(
                            "pgi, aip -> ag",
                            trial._rcholbara[iq, ik],
                            Ghalfa_reshape[:, ikpq, :, ik, :],
                            optimize=True,
                        )
                        + xp.einsum(
                            "pgi, bip -> bg",
                            trial._rcholbarb[iq, ik],
                            Ghalfb_reshape[:, ikpq, :, ik, :],
                            optimize=True,
                        )
                    )
                )
        synchronize()
        return vbias_plus, vbias_minus


def construct_force_bias_kptisdf_batch_single_det(
    hamiltonian: "KptISDF", walkers: "UHFWalkers", trial: "KptSingleDet", max_mem=4.0
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.
    trial : class
        Trial wavefunction object.
    handler : class
        MPIHandler instance.

    Returns
    -------
    vbias_plus : :class:`numpy.ndarray`
        Force bias for Lplus.
    vbias_minus : :class:`numpy.ndarray`
        Force bias for Lminus.
    """
    if walkers.rhf:
        if config.get_option("use_gpu"):
            nwalkers = walkers.nwalkers
            vbias_plus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            vbias_minus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            # ghalf shape: nwalkers, nk nup, hamiltonian.nk, nbsf
            Ghalfa_reshape = walkers.Ghalfa.reshape(
                nwalkers,
                hamiltonian.nk,
                trial.nalpha,
                hamiltonian.nk,
                hamiltonian.halfrot_cgto.shape[-1],
            )

            # slice Sset and Qplus according to memory
            mem_cost_Sset = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Sset)
                * hamiltonian.nk
                * (trial.nalpha)
                * 3
                * 16
                / (1024**3)
            )
            mem_cost_Qplus = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Qplus)
                * hamiltonian.nk
                * (trial.nalpha)
                * 4
                * 16
                / (1024**3)
            )

            num_nq_chunks_Sset = max(1, ceil(mem_cost_Sset / max_mem))
            nq_chunk_Sset_size = ceil(len(hamiltonian.Sset) / num_nq_chunks_Sset)
            nq_left = len(hamiltonian.Sset)
            handle = cutensornet.create()
            network_opts = NetworkOptions(handle=handle)
            if len(hamiltonian.Sset) > 0:
                for i in range(num_nq_chunks_Sset):
                    nq_chunk = min(nq_left, nq_chunk_Sset_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Sset[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    kpq_slice = hamiltonian.ikpq_mat[q_sls]
                    ga_kmq = slice_gf_kpq_k_qlis(
                        Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat
                    )  # q, k, w, p, r
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    X_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kmq,
                        options=network_opts,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    vbias_plus[
                        :, :, i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ] += 2.0j * cutensornet.contract(
                        "qwP, qPg -> wgq", X_wPa, L_q, options=network_opts
                    )

            num_nq_chunks_Qplus = max(1, ceil(mem_cost_Qplus / max_mem))
            nq_chunk_Qplus_size = ceil(len(hamiltonian.Qplus) / num_nq_chunks_Qplus)
            nq_left = len(hamiltonian.Qplus)
            if len(hamiltonian.Qplus) > 0:
                for i in range(num_nq_chunks_Qplus):
                    nq_chunk = min(nq_left, nq_chunk_Qplus_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Qplus[
                        i * nq_chunk_Qplus_size : i * nq_chunk_Qplus_size + nq_chunk
                    ]
                    kpq_slice = hamiltonian.ikpq_mat[q_sls]
                    ga_kmq = slice_gf_kpq_k_qlis(
                        Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat
                    )  # q, k, w, p, r
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    ga_kpq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikpq_mat)
                    rcgtoa_kpq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikpq_mat, q_sls)
                    X_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kmq,
                        options=network_opts,
                    )
                    Y_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kpq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kpq,
                        options=network_opts,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset)
                    ]
                    v1 = cutensornet.contract("qwP, qPg -> wgq", X_wPa, L_q, options=network_opts)
                    v2 = cutensornet.contract(
                        "qwP, qPg -> wgq", Y_wPa, L_q.conj(), options=network_opts
                    )
                    vbias_plus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        1j * xp.sqrt(2) * (v1 + v2)
                    )
                    vbias_minus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        1.0 * xp.sqrt(2) * (v1 - v2)
                    )
            cutensornet.destroy(handle)
            synchronize()
            return vbias_plus, vbias_minus
        else:
            pass
    else:
        if config.get_option("use_gpu"):
            nwalkers = walkers.nwalkers
            vbias_plus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            vbias_minus = xp.zeros(
                (nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            # ghalf shape: nwalkers, nk nup, hamiltonian.nk, nbsf
            Ghalfa_reshape = walkers.Ghalfa.reshape(
                nwalkers,
                hamiltonian.nk,
                trial.nalpha,
                hamiltonian.nk,
                hamiltonian.halfrot_cgto.shape[-1],
            )
            Ghalfb_reshape = walkers.Ghalfb.reshape(
                nwalkers,
                hamiltonian.nk,
                trial.nbeta,
                hamiltonian.nk,
                hamiltonian.halfrot_cgto.shape[-1],
            )

            # slice Sset and Qplus according to memory
            mem_cost_Sset = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Sset)
                * hamiltonian.nk
                * (trial.nalpha + trial.nbeta)
                * 16
                * 2
                / (1024**3)
            )
            mem_cost_Qplus = (
                max(nwalkers * hamiltonian.nbasis, hamiltonian.nisdf)
                * len(hamiltonian.Qplus)
                * hamiltonian.nk
                * (trial.nalpha + trial.nbeta)
                * 16
                * 2
                / (1024**3)
            )

            num_nq_chunks_Sset = max(1, ceil(mem_cost_Sset / max_mem))
            nq_chunk_Sset_size = ceil(len(hamiltonian.Sset) / num_nq_chunks_Sset)
            nq_left = len(hamiltonian.Sset)
            handle = cutensornet.create()
            if len(hamiltonian.Sset) > 0:
                for i in range(num_nq_chunks_Sset):
                    nq_chunk = min(nq_left, nq_chunk_Sset_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Sset[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    kpq_slice = hamiltonian.ikpq_mat[q_sls]
                    ga_kmq = slice_gf_kpq_k_qlis(
                        Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat
                    )  # q, k, w, p, r
                    gb_kmq = slice_gf_kpq_k_qlis(Ghalfb_reshape, q_sls, hamiltonian.ikmq_mat)
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    rcgtob_kmq = slice_cgto_kpq(trial._rcgtob, hamiltonian.ikmq_mat, q_sls)
                    network_opts = NetworkOptions(
                        handle=handle, memory_limit=0.8 * xp.cuda.Device().mem_info[0]
                    )
                    X_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kmq,
                        options=network_opts,
                    )
                    X_wPb = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtob_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        gb_kmq,
                        options=network_opts,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ]
                    vbias_plus[
                        :, :, i * nq_chunk_Sset_size : i * nq_chunk_Sset_size + nq_chunk
                    ] += 1j * cutensornet.contract(
                        "qwP, qPg -> wgq", X_wPa + X_wPb, L_q, options=network_opts
                    )

            num_nq_chunks_Qplus = max(1, ceil(mem_cost_Qplus / max_mem))
            nq_chunk_Qplus_size = ceil(len(hamiltonian.Qplus) / num_nq_chunks_Qplus)
            nq_left = len(hamiltonian.Qplus)
            if len(hamiltonian.Qplus) > 0:
                for i in range(num_nq_chunks_Qplus):
                    nq_chunk = min(nq_left, nq_chunk_Qplus_size)
                    nq_left -= nq_chunk
                    q_sls = hamiltonian.Qplus[
                        i * nq_chunk_Qplus_size : i * nq_chunk_Qplus_size + nq_chunk
                    ]
                    kpq_slice = hamiltonian.ikpq_mat[q_sls]
                    ga_kmq = slice_gf_kpq_k_qlis(
                        Ghalfa_reshape, q_sls, hamiltonian.ikmq_mat
                    )  # q, k, w, p, r
                    gb_kmq = slice_gf_kpq_k_qlis(Ghalfb_reshape, q_sls, hamiltonian.ikmq_mat)
                    rcgtoa_kmq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikmq_mat, q_sls)
                    rcgtob_kmq = slice_cgto_kpq(trial._rcgtob, hamiltonian.ikmq_mat, q_sls)
                    ga_kpq = slice_gf_kpq_k_qlis(Ghalfa_reshape, q_sls, hamiltonian.ikpq_mat)
                    gb_kpq = slice_gf_kpq_k_qlis(Ghalfb_reshape, q_sls, hamiltonian.ikpq_mat)
                    rcgtoa_kpq = slice_cgto_kpq(trial._rcgtoa, hamiltonian.ikpq_mat, q_sls)
                    rcgtob_kpq = slice_cgto_kpq(trial._rcgtob, hamiltonian.ikpq_mat, q_sls)
                    network_opts = NetworkOptions(
                        handle=handle, memory_limit=0.8 * xp.cuda.Device().mem_info[0]
                    )
                    X_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kmq,
                        options=network_opts,
                    )
                    X_wPb = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtob_kmq.conj(),
                        hamiltonian.halfrot_cgto,
                        gb_kmq,
                        options=network_opts,
                    )
                    Y_wPa = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtoa_kpq.conj(),
                        hamiltonian.halfrot_cgto,
                        ga_kpq,
                        options=network_opts,
                    )
                    Y_wPb = cutensornet.contract(
                        "qkPp, kPr, qkwpr -> qwP",
                        rcgtob_kpq.conj(),
                        hamiltonian.halfrot_cgto,
                        gb_kpq,
                        options=network_opts,
                    )
                    L_q = hamiltonian.cholM[
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset)
                    ]
                    v1 = cutensornet.contract(
                        "qwP, qPg -> wgq", X_wPa + X_wPb, L_q, options=network_opts
                    )
                    v2 = cutensornet.contract(
                        "qwP, qPg -> wgq", Y_wPa + Y_wPb, L_q.conj(), options=network_opts
                    )
                    vbias_plus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        0.5j * xp.sqrt(2) * (v1 + v2)
                    )
                    vbias_minus[
                        :,
                        :,
                        i * nq_chunk_Qplus_size
                        + len(hamiltonian.Sset) : i * nq_chunk_Qplus_size
                        + nq_chunk
                        + len(hamiltonian.Sset),
                    ] += (
                        0.5 * xp.sqrt(2) * (v1 - v2)
                    )
            cutensornet.destroy(handle)
            synchronize()
            return vbias_plus, vbias_minus
            # vbias_plus = xp.zeros((walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128)
            # vbias_minus = xp.zeros((walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128)
            # # ghalf shape: nwalkers, nk, nup, nk, nbsf
            # Ghalfa_reshape = walkers.Ghalfa.reshape(walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis)
            # Ghalfb_reshape = walkers.Ghalfb.reshape(walkers.nwalkers, hamiltonian.nk, trial.nbeta, hamiltonian.nk, hamiltonian.nbasis)
            # handle = cutensornet.create()
            # network_opts = NetworkOptions(handle=handle)
            # for iq in range(len(hamiltonian.Sset)):
            #     iq_real = hamiltonian.Sset[iq]
            #     ikpq = hamiltonian.ikpq_mat[iq_real]
            #     ga_kpq = slice_gf_kpq_k_given_q(Ghalfa_reshape, iq_real, hamiltonian.ikpq_mat)
            #     gb_kpq = slice_gf_kpq_k_given_q(Ghalfb_reshape, iq_real, hamiltonian.ikpq_mat)
            #     cgto_kpq = hamiltonian.halfrot_cgto[ikpq]
            #     X_wPa = contract_gf_cgto12_k_kpq(ga_kpq, trial._rcgtoa, cgto_kpq, iq_real, network_opts)
            #     X_wPb = contract_gf_cgto12_k_kpq(gb_kpq, trial._rcgtob, cgto_kpq, iq_real, network_opts)
            #     L_q = hamiltonian.cholM[iq]
            #     vbias_plus[:, :, iq] += 1j * (X_wPa + X_wPb).dot(L_q)
            #     vbias_minus[:, :, iq] += xp.zeros_like(vbias_plus[:, :, iq])

            # for iq in range(len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)):
            #     iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
            #     ikpq = hamiltonian.ikpq_mat[iq_real]
            #     cgto_kpq = hamiltonian.halfrot_cgto[ikpq]
            #     rcgtoa_kpq = trial._rcgtoa[ikpq]
            #     rcgtob_kpq = trial._rcgtob[ikpq]
            #     ga_kpq_k = slice_gf_kpq_k_given_q(Ghalfa_reshape, iq_real, hamiltonian.ikpq_mat)
            #     Y_wPa = contract_gf_cgto12_kpq_k(ga_kpq_k, rcgtoa_kpq, hamiltonian.halfrot_cgto, iq_real, network_opts)
            #     # del ga_kpq_k
            #     gb_kpq_k = slice_gf_kpq_k_given_q(Ghalfb_reshape, iq_real, hamiltonian.ikpq_mat)
            #     Y_wPb = contract_gf_cgto12_kpq_k(gb_kpq_k, rcgtob_kpq, hamiltonian.halfrot_cgto, iq_real, network_opts)
            #     # del gb_kpq_k
            #     ga_k_kpq = slice_gf_k_kpq_given_q(Ghalfa_reshape, iq_real, hamiltonian.ikpq_mat)
            #     X_wPa = contract_gf_cgto12_k_kpq(ga_k_kpq, trial._rcgtoa, cgto_kpq, iq_real, network_opts)
            #     # del ga_k_kpq
            #     gb_k_kpq = slice_gf_k_kpq_given_q(Ghalfb_reshape, iq_real, hamiltonian.ikpq_mat)
            #     X_wPb = contract_gf_cgto12_k_kpq(gb_k_kpq, trial._rcgtob, cgto_kpq, iq_real, network_opts)
            #     # del gb_k_kpq
            #     L_q = hamiltonian.cholM[iq]
            #     v1 = (X_wPa + X_wPb).dot(L_q)
            #     v2 = (Y_wPa + Y_wPb).dot(L_q.conj())
            #     vbias_plus[:, :, iq] += .5j * xp.sqrt(2) * (v1 + v2)
            #     vbias_minus[:, :, iq] += .5 * xp.sqrt(2) * (v1 - v2)
            # cutensornet.destroy(handle)
            # synchronize()
            # return vbias_plus, vbias_minus
        else:
            pass


def construct_force_bias_batch_single_det_chunked(hamiltonian, walkers, trial, handler):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.
    trial : class
        Trial wavefunction object.
    handler : class
        MPIHandler instance.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    assert hamiltonian.chunked
    assert xp.isrealobj(trial._rchola_chunk)

    Ghalfa = walkers.Ghalfa.reshape(walkers.nwalkers, walkers.nup * hamiltonian.nbasis)
    Ghalfb = walkers.Ghalfb.reshape(walkers.nwalkers, walkers.ndown * hamiltonian.nbasis)

    chol_idxs_chunk = hamiltonian.chol_idxs_chunk

    Ghalfa_recv = xp.zeros_like(Ghalfa)
    Ghalfb_recv = xp.zeros_like(Ghalfb)

    Ghalfa_send = Ghalfa.copy()
    Ghalfb_send = Ghalfb.copy()

    srank = handler.scomm.rank

    vbias_batch_real_recv = xp.zeros((hamiltonian.nchol, walkers.nwalkers))
    vbias_batch_imag_recv = xp.zeros((hamiltonian.nchol, walkers.nwalkers))

    vbias_batch_real_send = xp.zeros((hamiltonian.nchol, walkers.nwalkers))
    vbias_batch_imag_send = xp.zeros((hamiltonian.nchol, walkers.nwalkers))

    vbias_batch_real_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
        Ghalfa.T.real
    ) + trial._rcholb_chunk.dot(Ghalfb.T.real)
    vbias_batch_imag_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
        Ghalfa.T.imag
    ) + trial._rcholb_chunk.dot(Ghalfb.T.imag)

    receivers = handler.receivers
    for _ in range(handler.ssize - 1):
        synchronize()

        handler.scomm.Isend(Ghalfa_send, dest=receivers[srank], tag=1)
        handler.scomm.Isend(Ghalfb_send, dest=receivers[srank], tag=2)
        handler.scomm.Isend(vbias_batch_real_send, dest=receivers[srank], tag=3)
        handler.scomm.Isend(vbias_batch_imag_send, dest=receivers[srank], tag=4)

        idx = numpy.where(receivers == srank)[0]
        sender = int(idx.item())
        req1 = handler.scomm.Irecv(Ghalfa_recv, source=sender, tag=1)
        req2 = handler.scomm.Irecv(Ghalfb_recv, source=sender, tag=2)
        req3 = handler.scomm.Irecv(vbias_batch_real_recv, source=sender, tag=3)
        req4 = handler.scomm.Irecv(vbias_batch_imag_recv, source=sender, tag=4)
        req1.wait()
        req2.wait()
        req3.wait()
        req4.wait()

        handler.scomm.barrier()

        # prepare sending
        vbias_batch_real_send = vbias_batch_real_recv.copy()
        vbias_batch_imag_send = vbias_batch_imag_recv.copy()
        vbias_batch_real_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
            Ghalfa_recv.T.real
        ) + trial._rcholb_chunk.dot(Ghalfb_recv.T.real)
        vbias_batch_imag_send[chol_idxs_chunk, :] = trial._rchola_chunk.dot(
            Ghalfa_recv.T.imag
        ) + trial._rcholb_chunk.dot(Ghalfb_recv.T.imag)
        Ghalfa_send = Ghalfa_recv.copy()
        Ghalfb_send = Ghalfb_recv.copy()

    synchronize()
    handler.scomm.Isend(vbias_batch_real_send, dest=receivers[srank], tag=1)
    handler.scomm.Isend(vbias_batch_imag_send, dest=receivers[srank], tag=2)

    idx = numpy.where(receivers == srank)[0]
    sender = int(idx.item())
    req1 = handler.scomm.Irecv(vbias_batch_real_recv, source=sender, tag=1)
    req2 = handler.scomm.Irecv(vbias_batch_imag_recv, source=sender, tag=2)
    req1.wait()
    req2.wait()
    handler.scomm.barrier()

    vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
    vbias_batch.real = vbias_batch_real_recv.T.copy()
    vbias_batch.imag = vbias_batch_imag_recv.T.copy()
    synchronize()
    return vbias_batch


def construct_force_bias_batch_single_det_isdf_chunked(
    hamiltonian, walkers, rcgtoa, rcgtob, handler
):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.
    walkers : class
        walkers object.
    trial : class
        Trial wavefunction object.
    handler : class
        MPIHandler instance.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    assert hamiltonian.chunked
    assert xp.isrealobj(hamiltonian.cholM_chunk)

    Ghalfa = walkers.Ghalfa
    Ghalfb = walkers.Ghalfb

    chol_idxs_chunk = hamiltonian.chol_idxs_chunk

    Ghalfa_recv = xp.ascontiguousarray(xp.zeros_like(Ghalfa))
    Ghalfb_recv = xp.ascontiguousarray(xp.zeros_like(Ghalfb))

    Ghalfa_send = Ghalfa.copy()
    Ghalfb_send = Ghalfb.copy()

    srank = handler.scomm.rank

    vbias_batch_real_recv = xp.zeros((hamiltonian.nchol, walkers.nwalkers))
    vbias_batch_imag_recv = xp.zeros((hamiltonian.nchol, walkers.nwalkers))

    vbias_batch_real_send = xp.zeros((hamiltonian.nchol, walkers.nwalkers))
    vbias_batch_imag_send = xp.zeros((hamiltonian.nchol, walkers.nwalkers))

    handle = cutensornet.create()
    network_opts = NetworkOptions(handle=handle)
    vbias_batch_real_send[chol_idxs_chunk, :] = cutensornet.contract(
        "Pi, Pr, Pg, wir -> gw",
        rcgtoa,
        hamiltonian.cgto,
        hamiltonian.cholM_chunk,
        Ghalfa.real,
        options=network_opts,
    ) + cutensornet.contract(
        "Pi, Pr, Pg, wir -> gw",
        rcgtob,
        hamiltonian.cgto,
        hamiltonian.cholM_chunk,
        Ghalfb.real,
        options=network_opts,
    )
    vbias_batch_imag_send[chol_idxs_chunk, :] = cutensornet.contract(
        "Pi, Pr, Pg, wir -> gw",
        rcgtoa,
        hamiltonian.cgto,
        hamiltonian.cholM_chunk,
        Ghalfa.imag,
        options=network_opts,
    ) + cutensornet.contract(
        "Pi, Pr, Pg, wir -> gw",
        rcgtob,
        hamiltonian.cgto,
        hamiltonian.cholM_chunk,
        Ghalfb.imag,
        options=network_opts,
    )

    receivers = handler.receivers
    for _ in range(handler.ssize - 1):
        synchronize()

        handler.scomm.Isend(Ghalfa_send, dest=receivers[srank], tag=1)
        handler.scomm.Isend(Ghalfb_send, dest=receivers[srank], tag=2)
        handler.scomm.Isend(vbias_batch_real_send, dest=receivers[srank], tag=3)
        handler.scomm.Isend(vbias_batch_imag_send, dest=receivers[srank], tag=4)

        sender = numpy.where(receivers == srank)[0]
        req1 = handler.scomm.Irecv(Ghalfa_recv, source=sender, tag=1)
        req2 = handler.scomm.Irecv(Ghalfb_recv, source=sender, tag=2)
        req3 = handler.scomm.Irecv(vbias_batch_real_recv, source=sender, tag=3)
        req4 = handler.scomm.Irecv(vbias_batch_imag_recv, source=sender, tag=4)
        req1.wait()
        req2.wait()
        req3.wait()
        req4.wait()

        handler.scomm.barrier()

        # prepare sending
        vbias_batch_real_send = vbias_batch_real_recv.copy()
        vbias_batch_imag_send = vbias_batch_imag_recv.copy()
        vbias_batch_real_send[chol_idxs_chunk, :] = cutensornet.contract(
            "Pi, Pr, Pg, wir -> gw",
            rcgtoa,
            hamiltonian.cgto,
            hamiltonian.cholM_chunk,
            Ghalfa_recv.real,
            options=network_opts,
        ) + cutensornet.contract(
            "Pi, Pr, Pg, wir -> gw",
            rcgtob,
            hamiltonian.cgto,
            hamiltonian.cholM_chunk,
            Ghalfb_recv.real,
            options=network_opts,
        )
        vbias_batch_imag_send[chol_idxs_chunk, :] = cutensornet.contract(
            "Pi, Pr, Pg, wir -> gw",
            rcgtoa,
            hamiltonian.cgto,
            hamiltonian.cholM_chunk,
            Ghalfa_recv.imag,
            options=network_opts,
        ) + cutensornet.contract(
            "Pi, Pr, Pg, wir -> gw",
            rcgtob,
            hamiltonian.cgto,
            hamiltonian.cholM_chunk,
            Ghalfb_recv.imag,
            options=network_opts,
        )
        Ghalfa_send = Ghalfa_recv.copy()
        Ghalfb_send = Ghalfb_recv.copy()

    synchronize()
    handler.scomm.Isend(vbias_batch_real_send, dest=receivers[srank], tag=1)
    handler.scomm.Isend(vbias_batch_imag_send, dest=receivers[srank], tag=2)

    sender = numpy.where(receivers == srank)[0]
    req1 = handler.scomm.Irecv(vbias_batch_real_recv, source=sender, tag=1)
    req2 = handler.scomm.Irecv(vbias_batch_imag_recv, source=sender, tag=2)
    req1.wait()
    req2.wait()
    handler.scomm.barrier()

    vbias_batch = xp.empty((walkers.nwalkers, hamiltonian.nchol), dtype=Ghalfa.dtype)
    vbias_batch.real = vbias_batch_real_recv.T.copy()
    vbias_batch.imag = vbias_batch_imag_recv.T.copy()
    synchronize()
    return vbias_batch


def construct_force_bias_kptsymm_batch_single_det_chunked(hamiltonian, walkers, trial, handler):
    """Compute optimal force bias.

    Uses rotated Green's function.

    Parameters
    ----------
    hamiltonian : class
        hamiltonian object.

    walkers : class
        walkers object.

    trial : class
        Trial wavefunction object.

    Returns
    -------
    xbar : :class:`numpy.ndarray`
        Force bias.
    """
    assert hamiltonian.chunked
    if walkers.rhf:
        Ghalfa = walkers.Ghalfa.reshape(
            walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
        )

        chol_idxs_chunk = hamiltonian.chol_idxs_chunk

        Ghalfa_recv = xp.zeros_like(Ghalfa)
        Ghalfa_send = Ghalfa.copy()

        srank = handler.scomm.rank

        vbias_batch_plus_recv = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )
        vbias_batch_minus_recv = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )

        vbias_batch_plus_send = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )
        vbias_batch_minus_send = xp.zeros(
            (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
        )

        if config.get_option("use_gpu"):
            if len(hamiltonian.Sset) > 0:
                iSset = xp.arange(len(hamiltonian.Sset))
                ik_Sset = (
                    xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Sset))
                    .reshape(hamiltonian.nk, len(hamiltonian.Sset))
                    .T
                )
                ikpq_S = hamiltonian.ikpq_mat[hamiltonian.Sset]
                Gk_kpq = Ghalfa[:, ik_Sset, :, ikpq_S, :]
                Gkpq_k = Ghalfa[:, ikpq_S, :, ik_Sset, :]
                tmp1 = xp.einsum(
                    "qkigp, qkaip -> agq", trial._rchola_chunk[iSset], Gk_kpq, optimize=True
                )
                tmp2 = xp.einsum(
                    "qkpgi, qkaip -> agq", trial._rcholbara_chunk[iSset], Gkpq_k, optimize=True
                )
                vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iSset] += 1.0j * (tmp1 + tmp2)
                vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iSset] += tmp1 - tmp2

            if len(hamiltonian.Qplus) > 0:
                iQplus = xp.arange(len(hamiltonian.Qplus))
                ik_Qplus = (
                    xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Qplus))
                    .reshape(hamiltonian.nk, len(hamiltonian.Qplus))
                    .T
                )
                ikpq_Q = hamiltonian.ikpq_mat[hamiltonian.Qplus]
                Gk_kpq = Ghalfa[:, ik_Qplus, :, ikpq_Q, :]
                Gkpq_k = Ghalfa[:, ikpq_Q, :, ik_Qplus, :]
                iQplus_real = iQplus + len(hamiltonian.Sset)
                tmp1 = xp.einsum(
                    "qkigp, qkaip -> agq", trial._rchola_chunk[iQplus_real], Gk_kpq, optimize=True
                )
                tmp2 = xp.einsum(
                    "qkpgi, qkaip -> agq",
                    trial._rcholbara_chunk[iQplus_real],
                    Gkpq_k,
                    optimize=True,
                )
                vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iQplus_real] += (
                    1.0j * xp.sqrt(2) * (tmp1 + tmp2)
                )
                vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iQplus_real] += xp.sqrt(2) * (
                    tmp1 - tmp2
                )
        else:
            for iq in range(len(hamiltonian.Sset)):
                iq_real = hamiltonian.Sset[iq]
                for ik in range(hamiltonian.nk):
                    ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                    vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 1.0j * xp.einsum(
                        "igp, aip -> ag",
                        trial._rchola_chunk[iq, ik],
                        Ghalfa[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 1.0j * xp.einsum(
                        "pgi, aip -> ag",
                        trial._rcholbara_chunk[iq, ik],
                        Ghalfa[:, ikpq, :, ik, :],
                        optimize=True,
                    )

                    vbias_batch_minus_send[:, chol_idxs_chunk, iq] += xp.einsum(
                        "igp, aip -> ag",
                        trial._rchola_chunk[iq, ik],
                        Ghalfa[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= xp.einsum(
                        "pgi, aip -> ag",
                        trial._rcholbara_chunk[iq, ik],
                        Ghalfa[:, ikpq, :, ik, :],
                        optimize=True,
                    )

            for iq in range(len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)):
                iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
                for ik in range(hamiltonian.nk):
                    ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                    vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                        1.0j
                        * xp.sqrt(2)
                        * xp.einsum(
                            "igp, aip -> ag",
                            trial._rchola_chunk[iq, ik],
                            Ghalfa[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                        1.0j
                        * xp.sqrt(2)
                        * xp.einsum(
                            "pgi, aip -> ag",
                            trial._rcholbara_chunk[iq, ik],
                            Ghalfa[:, ikpq, :, ik, :],
                            optimize=True,
                        )
                    )

                    vbias_batch_minus_send[:, chol_idxs_chunk, iq] += xp.sqrt(2) * xp.einsum(
                        "igp, aip -> ag",
                        trial._rchola_chunk[iq, ik],
                        Ghalfa[:, ik, :, ikpq, :],
                        optimize=True,
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= xp.sqrt(2) * xp.einsum(
                        "pgi, aip -> ag",
                        trial._rcholbara_chunk[iq, ik],
                        Ghalfa[:, ikpq, :, ik, :],
                        optimize=True,
                    )

        receivers = handler.receivers
        for _ in range(handler.ssize - 1):
            synchronize()

            handler.scomm.Isend(Ghalfa_send, dest=receivers[srank], tag=1)
            handler.scomm.Isend(vbias_batch_plus_send, dest=receivers[srank], tag=2)
            handler.scomm.Isend(vbias_batch_minus_send, dest=receivers[srank], tag=3)

            sender = numpy.where(receivers == srank)[0]
            req1 = handler.scomm.Irecv(Ghalfa_recv, source=sender, tag=1)
            req2 = handler.scomm.Irecv(vbias_batch_plus_recv, source=sender, tag=2)
            req3 = handler.scomm.Irecv(vbias_batch_minus_recv, source=sender, tag=3)
            req1.wait()
            req2.wait()
            req3.wait()

            handler.scomm.barrier()

            # prepare sending
            vbias_batch_plus_send = vbias_batch_plus_recv.copy()
            vbias_batch_minus_send = vbias_batch_minus_recv.copy()
            if config.get_option("use_gpu"):
                if len(hamiltonian.Sset) > 0:
                    iSset = xp.arange(len(hamiltonian.Sset))
                    ik_Sset = (
                        xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Sset))
                        .reshape(hamiltonian.nk, len(hamiltonian.Sset))
                        .T
                    )
                    ikpq_S = hamiltonian.ikpq_mat[hamiltonian.Sset]
                    Gk_kpq = Ghalfa_recv[:, ik_Sset, :, ikpq_S, :]
                    Gkpq_k = Ghalfa_recv[:, ikpq_S, :, ik_Sset, :]
                    tmp1 = xp.einsum(
                        "qkigp, qkaip -> agq", trial._rchola_chunk[iSset], Gk_kpq, optimize=True
                    )
                    tmp2 = xp.einsum(
                        "qkpgi, qkaip -> agq", trial._rcholbara_chunk[iSset], Gkpq_k, optimize=True
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iSset] += 1.0j * (
                        tmp1 + tmp2
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iSset] += tmp1 - tmp2

                if len(hamiltonian.Qplus) > 0:
                    iQplus = xp.arange(len(hamiltonian.Qplus))
                    ik_Qplus = (
                        xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Qplus))
                        .reshape(hamiltonian.nk, len(hamiltonian.Qplus))
                        .T
                    )
                    ikpq_Q = hamiltonian.ikpq_mat[hamiltonian.Qplus]
                    Gk_kpq = Ghalfa_recv[:, ik_Qplus, :, ikpq_Q, :]
                    Gkpq_k = Ghalfa_recv[:, ikpq_Q, :, ik_Qplus, :]
                    iQplus_real = iQplus + len(hamiltonian.Sset)
                    tmp1 = xp.einsum(
                        "qkigp, qkaip -> agq",
                        trial._rchola_chunk[iQplus_real],
                        Gk_kpq,
                        optimize=True,
                    )
                    tmp2 = xp.einsum(
                        "qkpgi, qkaip -> agq",
                        trial._rcholbara_chunk[iQplus_real],
                        Gkpq_k,
                        optimize=True,
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iQplus_real] += (
                        1.0j * xp.sqrt(2) * (tmp1 + tmp2)
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iQplus_real] += xp.sqrt(
                        2
                    ) * (tmp1 - tmp2)
            else:
                for iq in range(len(hamiltonian.Sset)):
                    iq_real = hamiltonian.Sset[iq]
                    for ik in range(hamiltonian.nk):
                        ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 1.0j * xp.einsum(
                            "igp, aip -> ag",
                            trial._rchola_chunk[iq, ik],
                            Ghalfa_recv[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 1.0j * xp.einsum(
                            "pgi, aip -> ag",
                            trial._rcholbara_chunk[iq, ik],
                            Ghalfa_recv[:, ikpq, :, ik, :],
                            optimize=True,
                        )

                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] += xp.einsum(
                            "igp, aip -> ag",
                            trial._rchola_chunk[iq, ik],
                            Ghalfa_recv[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= xp.einsum(
                            "pgi, aip -> ag",
                            trial._rcholbara_chunk[iq, ik],
                            Ghalfa_recv[:, ikpq, :, ik, :],
                            optimize=True,
                        )

                for iq in range(
                    len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)
                ):
                    iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
                    for ik in range(hamiltonian.nk):
                        ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                            1.0j
                            * xp.sqrt(2)
                            * xp.einsum(
                                "igp, aip -> ag",
                                trial._rchola_chunk[iq, ik],
                                Ghalfa_recv[:, ik, :, ikpq, :],
                                optimize=True,
                            )
                        )
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                            1.0j
                            * xp.sqrt(2)
                            * xp.einsum(
                                "pgi, aip -> ag",
                                trial._rcholbara_chunk[iq, ik],
                                Ghalfa_recv[:, ikpq, :, ik, :],
                                optimize=True,
                            )
                        )

                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] += xp.sqrt(2) * xp.einsum(
                            "igp, aip -> ag",
                            trial._rchola_chunk[iq, ik],
                            Ghalfa_recv[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= xp.sqrt(2) * xp.einsum(
                            "pgi, aip -> ag",
                            trial._rcholbara_chunk[iq, ik],
                            Ghalfa_recv[:, ikpq, :, ik, :],
                            optimize=True,
                        )
            Ghalfa_send = Ghalfa_recv.copy()

        synchronize()
        handler.scomm.Isend(vbias_batch_plus_send, dest=receivers[srank], tag=1)
        handler.scomm.Isend(vbias_batch_minus_send, dest=receivers[srank], tag=2)

        sender = numpy.where(receivers == srank)[0]
        req1 = handler.scomm.Irecv(vbias_batch_plus_recv, source=sender, tag=1)
        req2 = handler.scomm.Irecv(vbias_batch_minus_recv, source=sender, tag=2)
        req1.wait()
        req2.wait()
        handler.scomm.barrier()

        vbias_plus = vbias_batch_plus_recv.copy()
        vbias_minus = vbias_batch_minus_recv.copy()
        synchronize()
    else:
        if trial.nbeta > 0:
            Ghalfa = walkers.Ghalfa.reshape(
                walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
            )
            Ghalfb = walkers.Ghalfb.reshape(
                walkers.nwalkers, hamiltonian.nk, trial.nbeta, hamiltonian.nk, hamiltonian.nbasis
            )

            chol_idxs_chunk = hamiltonian.chol_idxs_chunk

            Ghalfa_recv = xp.zeros_like(Ghalfa)
            Ghalfb_recv = xp.zeros_like(Ghalfb)

            Ghalfa_send = Ghalfa.copy()
            Ghalfb_send = Ghalfb.copy()

            srank = handler.scomm.rank

            vbias_batch_plus_recv = xp.zeros(
                (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            vbias_batch_minus_recv = xp.zeros(
                (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )

            vbias_batch_plus_send = xp.zeros(
                (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            vbias_batch_minus_send = xp.zeros(
                (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )

            if config.get_option("use_gpu"):
                if len(hamiltonian.Sset) > 0:
                    iSset = xp.arange(len(hamiltonian.Sset))
                    ik_Sset = (
                        xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Sset))
                        .reshape(hamiltonian.nk, len(hamiltonian.Sset))
                        .T
                    )
                    ikpq_S = hamiltonian.ikpq_mat[hamiltonian.Sset]
                    Gak_kpq = Ghalfa[:, ik_Sset, :, ikpq_S, :]
                    Gakpq_k = Ghalfa[:, ikpq_S, :, ik_Sset, :]
                    Gbk_kpq = Ghalfb[:, ik_Sset, :, ikpq_S, :]
                    Gbkpq_k = Ghalfb[:, ikpq_S, :, ik_Sset, :]
                    tmp1 = xp.einsum(
                        "qkigp, qkaip -> agq", trial._rchola_chunk[iSset], Gak_kpq, optimize=True
                    ) + xp.einsum(
                        "qkigp, qkaip -> agq", trial._rcholb_chunk[iSset], Gbk_kpq, optimize=True
                    )
                    tmp2 = xp.einsum(
                        "qkpgi, qkaip -> agq", trial._rcholbara_chunk[iSset], Gakpq_k, optimize=True
                    ) + xp.einsum(
                        "qkpgi, qkaip -> agq", trial._rcholbarb_chunk[iSset], Gbkpq_k, optimize=True
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iSset] += 0.5j * (
                        tmp1 + tmp2
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iSset] += 0.5 * (
                        tmp1 - tmp2
                    )

                if len(hamiltonian.Qplus) > 0:
                    iQplus = xp.arange(len(hamiltonian.Qplus))
                    ik_Qplus = (
                        xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Qplus))
                        .reshape(hamiltonian.nk, len(hamiltonian.Qplus))
                        .T
                    )
                    ikpq_Q = hamiltonian.ikpq_mat[hamiltonian.Qplus]
                    Gak_kpq = Ghalfa[:, ik_Qplus, :, ikpq_Q, :]
                    Gakpq_k = Ghalfa[:, ikpq_Q, :, ik_Qplus, :]
                    Gbk_kpq = Ghalfb[:, ik_Qplus, :, ikpq_Q, :]
                    Gbkpq_k = Ghalfb[:, ikpq_Q, :, ik_Qplus, :]
                    iQplus_real = iQplus + len(hamiltonian.Sset)
                    tmp1 = xp.einsum(
                        "qkigp, qkaip -> agq",
                        trial._rchola_chunk[iQplus_real],
                        Gak_kpq,
                        optimize=True,
                    ) + xp.einsum(
                        "qkigp, qkaip -> agq",
                        trial._rcholb_chunk[iQplus_real],
                        Gbk_kpq,
                        optimize=True,
                    )
                    tmp2 = xp.einsum(
                        "qkpgi, qkaip -> agq",
                        trial._rcholbara_chunk[iQplus_real],
                        Gakpq_k,
                        optimize=True,
                    ) + xp.einsum(
                        "qkpgi, qkaip -> agq",
                        trial._rcholbarb_chunk[iQplus_real],
                        Gbkpq_k,
                        optimize=True,
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iQplus_real] += (
                        0.5j * xp.sqrt(2) * (tmp1 + tmp2)
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iQplus_real] += (
                        0.5 * xp.sqrt(2) * (tmp1 - tmp2)
                    )
            else:
                for iq in range(len(hamiltonian.Sset)):
                    iq_real = hamiltonian.Sset[iq]
                    for ik in range(hamiltonian.nk):
                        ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 0.5j * (
                            xp.einsum(
                                "igp, aip -> ag",
                                trial._rchola_chunk[iq, ik],
                                Ghalfa[:, ik, :, ikpq, :],
                                optimize=True,
                            )
                            + xp.einsum(
                                "igp, bip -> bg",
                                trial._rcholb_chunk[iq, ik],
                                Ghalfb[:, ik, :, ikpq, :],
                                optimize=True,
                            )
                        )
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 0.5j * (
                            xp.einsum(
                                "pgi, aip -> ag",
                                trial._rcholbara_chunk[iq, ik],
                                Ghalfa[:, ikpq, :, ik, :],
                                optimize=True,
                            )
                            + xp.einsum(
                                "pgi, bip -> bg",
                                trial._rcholbarb_chunk[iq, ik],
                                Ghalfb[:, ikpq, :, ik, :],
                                optimize=True,
                            )
                        )

                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] += 0.5 * (
                            xp.einsum(
                                "igp, aip -> ag",
                                trial._rchola_chunk[iq, ik],
                                Ghalfa[:, ik, :, ikpq, :],
                                optimize=True,
                            )
                            + xp.einsum(
                                "igp, bip -> bg",
                                trial._rcholb_chunk[iq, ik],
                                Ghalfb[:, ik, :, ikpq, :],
                                optimize=True,
                            )
                        )
                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= 0.5 * (
                            xp.einsum(
                                "pgi, aip -> ag",
                                trial._rcholbara_chunk[iq, ik],
                                Ghalfa[:, ikpq, :, ik, :],
                                optimize=True,
                            )
                            + xp.einsum(
                                "pgi, bip -> bg",
                                trial._rcholbarb_chunk[iq, ik],
                                Ghalfb[:, ikpq, :, ik, :],
                                optimize=True,
                            )
                        )

                for iq in range(
                    len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)
                ):
                    iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
                    for ik in range(hamiltonian.nk):
                        ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                            0.5j
                            * xp.sqrt(2)
                            * (
                                xp.einsum(
                                    "igp, aip -> ag",
                                    trial._rchola_chunk[iq, ik],
                                    Ghalfa[:, ik, :, ikpq, :],
                                    optimize=True,
                                )
                                + xp.einsum(
                                    "igp, bip -> bg",
                                    trial._rcholb_chunk[iq, ik],
                                    Ghalfb[:, ik, :, ikpq, :],
                                    optimize=True,
                                )
                            )
                        )
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                            0.5j
                            * xp.sqrt(2)
                            * (
                                xp.einsum(
                                    "pgi, aip -> ag",
                                    trial._rcholbara_chunk[iq, ik],
                                    Ghalfa[:, ikpq, :, ik, :],
                                    optimize=True,
                                )
                                + xp.einsum(
                                    "pgi, bip -> bg",
                                    trial._rcholbarb_chunk[iq, ik],
                                    Ghalfb[:, ikpq, :, ik, :],
                                    optimize=True,
                                )
                            )
                        )

                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] += (
                            0.5
                            * xp.sqrt(2)
                            * (
                                xp.einsum(
                                    "igp, aip -> ag",
                                    trial._rchola_chunk[iq, ik],
                                    Ghalfa[:, ik, :, ikpq, :],
                                    optimize=True,
                                )
                                + xp.einsum(
                                    "igp, bip -> bg",
                                    trial._rcholb_chunk[iq, ik],
                                    Ghalfb[:, ik, :, ikpq, :],
                                    optimize=True,
                                )
                            )
                        )
                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= (
                            0.5
                            * xp.sqrt(2)
                            * (
                                xp.einsum(
                                    "pgi, aip -> ag",
                                    trial._rcholbara_chunk[iq, ik],
                                    Ghalfa[:, ikpq, :, ik, :],
                                    optimize=True,
                                )
                                + xp.einsum(
                                    "pgi, bip -> bg",
                                    trial._rcholbarb_chunk[iq, ik],
                                    Ghalfb[:, ikpq, :, ik, :],
                                    optimize=True,
                                )
                            )
                        )

            receivers = handler.receivers
            for _ in range(handler.ssize - 1):
                synchronize()

                handler.scomm.Isend(Ghalfa_send, dest=receivers[srank], tag=1)
                handler.scomm.Isend(Ghalfb_send, dest=receivers[srank], tag=2)
                handler.scomm.Isend(vbias_batch_plus_send, dest=receivers[srank], tag=3)
                handler.scomm.Isend(vbias_batch_minus_send, dest=receivers[srank], tag=4)

                sender = numpy.where(receivers == srank)[0]
                req1 = handler.scomm.Irecv(Ghalfa_recv, source=sender, tag=1)
                req2 = handler.scomm.Irecv(Ghalfb_recv, source=sender, tag=2)
                req3 = handler.scomm.Irecv(vbias_batch_plus_recv, source=sender, tag=3)
                req4 = handler.scomm.Irecv(vbias_batch_minus_recv, source=sender, tag=4)
                req1.wait()
                req2.wait()
                req3.wait()
                req4.wait()

                handler.scomm.barrier()

                # prepare sending
                vbias_batch_plus_send = vbias_batch_plus_recv.copy()
                vbias_batch_minus_send = vbias_batch_minus_recv.copy()
                if len(hamiltonian.Sset) > 0:
                    iSset = xp.arange(len(hamiltonian.Sset))
                    ik_Sset = (
                        xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Sset))
                        .reshape(hamiltonian.nk, len(hamiltonian.Sset))
                        .T
                    )
                    ikpq_S = hamiltonian.ikpq_mat[hamiltonian.Sset]
                    Gak_kpq = Ghalfa_recv[:, ik_Sset, :, ikpq_S, :]
                    Gakpq_k = Ghalfa_recv[:, ikpq_S, :, ik_Sset, :]
                    Gbk_kpq = Ghalfb_recv[:, ik_Sset, :, ikpq_S, :]
                    Gbkpq_k = Ghalfb_recv[:, ikpq_S, :, ik_Sset, :]
                    tmp1 = xp.einsum(
                        "qkigp, qkaip -> agq", trial._rchola_chunk[iSset], Gak_kpq, optimize=True
                    ) + xp.einsum(
                        "qkigp, qkaip -> agq", trial._rcholb_chunk[iSset], Gbk_kpq, optimize=True
                    )
                    tmp2 = xp.einsum(
                        "qkpgi, qkaip -> agq", trial._rcholbara_chunk[iSset], Gakpq_k, optimize=True
                    ) + xp.einsum(
                        "qkpgi, qkaip -> agq", trial._rcholbarb_chunk[iSset], Gbkpq_k, optimize=True
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iSset] += 0.5j * (
                        tmp1 + tmp2
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iSset] += 0.5 * (
                        tmp1 - tmp2
                    )

                if len(hamiltonian.Qplus) > 0:
                    iQplus = xp.arange(len(hamiltonian.Qplus))
                    ik_Qplus = (
                        xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Qplus))
                        .reshape(hamiltonian.nk, len(hamiltonian.Qplus))
                        .T
                    )
                    ikpq_Q = hamiltonian.ikpq_mat[hamiltonian.Qplus]
                    Gak_kpq = Ghalfa_recv[:, ik_Qplus, :, ikpq_Q, :]
                    Gakpq_k = Ghalfa_recv[:, ikpq_Q, :, ik_Qplus, :]
                    Gbk_kpq = Ghalfb_recv[:, ik_Qplus, :, ikpq_Q, :]
                    Gbkpq_k = Ghalfb_recv[:, ikpq_Q, :, ik_Qplus, :]
                    iQplus_real = iQplus + len(hamiltonian.Sset)
                    tmp1 = xp.einsum(
                        "qkigp, qkaip -> agq",
                        trial._rchola_chunk[iQplus_real],
                        Gak_kpq,
                        optimize=True,
                    ) + xp.einsum(
                        "qkigp, qkaip -> agq",
                        trial._rcholb_chunk[iQplus_real],
                        Gbk_kpq,
                        optimize=True,
                    )
                    tmp2 = xp.einsum(
                        "qkpgi, qkaip -> agq",
                        trial._rcholbara_chunk[iQplus_real],
                        Gakpq_k,
                        optimize=True,
                    ) + xp.einsum(
                        "qkpgi, qkaip -> agq",
                        trial._rcholbarb_chunk[iQplus_real],
                        Gbkpq_k,
                        optimize=True,
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iQplus_real] += (
                        0.5j * xp.sqrt(2) * (tmp1 + tmp2)
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iQplus_real] += (
                        0.5 * xp.sqrt(2) * (tmp1 - tmp2)
                    )

                else:
                    for iq in range(len(hamiltonian.Sset)):
                        iq_real = hamiltonian.Sset[iq]
                        for ik in range(hamiltonian.nk):
                            ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                            vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 0.5j * (
                                xp.einsum(
                                    "igp, aip -> ag",
                                    trial._rchola_chunk[iq, ik],
                                    Ghalfa_recv[:, ik, :, ikpq, :],
                                    optimize=True,
                                )
                                + xp.einsum(
                                    "igp, bip -> bg",
                                    trial._rcholb_chunk[iq, ik],
                                    Ghalfb_recv[:, ik, :, ikpq, :],
                                    optimize=True,
                                )
                            )
                            vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 0.5j * (
                                xp.einsum(
                                    "pgi, aip -> ag",
                                    trial._rcholbara_chunk[iq, ik],
                                    Ghalfa_recv[:, ikpq, :, ik, :],
                                    optimize=True,
                                )
                                + xp.einsum(
                                    "pgi, bip -> bg",
                                    trial._rcholbarb_chunk[iq, ik],
                                    Ghalfb_recv[:, ikpq, :, ik, :],
                                    optimize=True,
                                )
                            )

                            vbias_batch_minus_send[:, chol_idxs_chunk, iq] += 0.5 * (
                                xp.einsum(
                                    "igp, aip -> ag",
                                    trial._rchola_chunk[iq, ik],
                                    Ghalfa_recv[:, ik, :, ikpq, :],
                                    optimize=True,
                                )
                                + xp.einsum(
                                    "igp, bip -> bg",
                                    trial._rcholb_chunk[iq, ik],
                                    Ghalfb_recv[:, ik, :, ikpq, :],
                                    optimize=True,
                                )
                            )
                            vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= 0.5 * (
                                xp.einsum(
                                    "pgi, aip -> ag",
                                    trial._rcholbara_chunk[iq, ik],
                                    Ghalfa_recv[:, ikpq, :, ik, :],
                                    optimize=True,
                                )
                                + xp.einsum(
                                    "pgi, bip -> bg",
                                    trial._rcholbarb_chunk[iq, ik],
                                    Ghalfb_recv[:, ikpq, :, ik, :],
                                    optimize=True,
                                )
                            )

                    for iq in range(
                        len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)
                    ):
                        iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
                        for ik in range(hamiltonian.nk):
                            ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                            vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                                0.5j
                                * xp.sqrt(2)
                                * (
                                    xp.einsum(
                                        "igp, aip -> ag",
                                        trial._rchola_chunk[iq, ik],
                                        Ghalfa_recv[:, ik, :, ikpq, :],
                                        optimize=True,
                                    )
                                    + xp.einsum(
                                        "igp, bip -> bg",
                                        trial._rcholb_chunk[iq, ik],
                                        Ghalfb_recv[:, ik, :, ikpq, :],
                                        optimize=True,
                                    )
                                )
                            )
                            vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                                0.5j
                                * xp.sqrt(2)
                                * (
                                    xp.einsum(
                                        "pgi, aip -> ag",
                                        trial._rcholbara_chunk[iq, ik],
                                        Ghalfa_recv[:, ikpq, :, ik, :],
                                        optimize=True,
                                    )
                                    + xp.einsum(
                                        "pgi, bip -> bg",
                                        trial._rcholbarb_chunk[iq, ik],
                                        Ghalfb_recv[:, ikpq, :, ik, :],
                                        optimize=True,
                                    )
                                )
                            )

                            vbias_batch_minus_send[:, chol_idxs_chunk, iq] += (
                                0.5
                                * xp.sqrt(2)
                                * (
                                    xp.einsum(
                                        "igp, aip -> ag",
                                        trial._rchola_chunk[iq, ik],
                                        Ghalfa_recv[:, ik, :, ikpq, :],
                                        optimize=True,
                                    )
                                    + xp.einsum(
                                        "igp, bip -> bg",
                                        trial._rcholb_chunk[iq, ik],
                                        Ghalfb_recv[:, ik, :, ikpq, :],
                                        optimize=True,
                                    )
                                )
                            )
                            vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= (
                                0.5
                                * xp.sqrt(2)
                                * (
                                    xp.einsum(
                                        "pgi, aip -> ag",
                                        trial._rcholbara_chunk[iq, ik],
                                        Ghalfa_recv[:, ikpq, :, ik, :],
                                        optimize=True,
                                    )
                                    + xp.einsum(
                                        "pgi, bip -> bg",
                                        trial._rcholbarb_chunk[iq, ik],
                                        Ghalfb_recv[:, ikpq, :, ik, :],
                                        optimize=True,
                                    )
                                )
                            )
                Ghalfa_send = Ghalfa_recv.copy()
                Ghalfb_send = Ghalfb_recv.copy()

            synchronize()
            handler.scomm.Isend(vbias_batch_plus_send, dest=receivers[srank], tag=1)
            handler.scomm.Isend(vbias_batch_minus_send, dest=receivers[srank], tag=2)

            sender = numpy.where(receivers == srank)[0]
            req1 = handler.scomm.Irecv(vbias_batch_plus_recv, source=sender, tag=1)
            req2 = handler.scomm.Irecv(vbias_batch_minus_recv, source=sender, tag=2)
            req1.wait()
            req2.wait()
            handler.scomm.barrier()

            vbias_plus = vbias_batch_plus_recv.copy()
            vbias_minus = vbias_batch_minus_recv.copy()
            synchronize()
        else:
            Ghalfa = walkers.Ghalfa.reshape(
                walkers.nwalkers, hamiltonian.nk, trial.nalpha, hamiltonian.nk, hamiltonian.nbasis
            )

            chol_idxs_chunk = hamiltonian.chol_idxs_chunk

            Ghalfa_recv = xp.zeros_like(Ghalfa)

            Ghalfa_send = Ghalfa.copy()

            srank = handler.scomm.rank

            vbias_batch_plus_recv = xp.zeros(
                (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            vbias_batch_minus_recv = xp.zeros(
                (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )

            vbias_batch_plus_send = xp.zeros(
                (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )
            vbias_batch_minus_send = xp.zeros(
                (walkers.nwalkers, hamiltonian.nchol, hamiltonian.unique_nk), dtype=numpy.complex128
            )

            if config.get_option("use_gpu"):
                if len(hamiltonian.Sset) > 0:
                    iSset = xp.arange(len(hamiltonian.Sset))
                    ik_Sset = (
                        xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Sset))
                        .reshape(hamiltonian.nk, len(hamiltonian.Sset))
                        .T
                    )
                    ikpq_S = hamiltonian.ikpq_mat[hamiltonian.Sset]
                    Gak_kpq = Ghalfa[:, ik_Sset, :, ikpq_S, :]
                    Gakpq_k = Ghalfa[:, ikpq_S, :, ik_Sset, :]
                    tmp1 = xp.einsum(
                        "qkigp, qkaip -> agq", trial._rchola_chunk[iSset], Gak_kpq, optimize=True
                    )
                    tmp2 = xp.einsum(
                        "qkpgi, qkaip -> agq", trial._rcholbara_chunk[iSset], Gakpq_k, optimize=True
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iSset] += 0.5j * (
                        tmp1 + tmp2
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iSset] += 0.5 * (
                        tmp1 - tmp2
                    )

                if len(hamiltonian.Qplus) > 0:
                    iQplus = xp.arange(len(hamiltonian.Qplus))
                    ik_Qplus = (
                        xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Qplus))
                        .reshape(hamiltonian.nk, len(hamiltonian.Qplus))
                        .T
                    )
                    ikpq_Q = hamiltonian.ikpq_mat[hamiltonian.Qplus]
                    Gak_kpq = Ghalfa[:, ik_Qplus, :, ikpq_Q, :]
                    Gakpq_k = Ghalfa[:, ikpq_Q, :, ik_Qplus, :]
                    iQplus_real = iQplus + len(hamiltonian.Sset)
                    tmp1 = xp.einsum(
                        "qkigp, qkaip -> agq",
                        trial._rchola_chunk[iQplus_real],
                        Gak_kpq,
                        optimize=True,
                    )
                    tmp2 = xp.einsum(
                        "qkpgi, qkaip -> agq",
                        trial._rcholbara_chunk[iQplus_real],
                        Gakpq_k,
                        optimize=True,
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iQplus_real] += (
                        0.5j * xp.sqrt(2) * (tmp1 + tmp2)
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iQplus_real] += (
                        0.5 * xp.sqrt(2) * (tmp1 - tmp2)
                    )
            else:
                for iq in range(len(hamiltonian.Sset)):
                    iq_real = hamiltonian.Sset[iq]
                    for ik in range(hamiltonian.nk):
                        ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 0.5j * xp.einsum(
                            "igp, aip -> ag",
                            trial._rchola_chunk[iq, ik],
                            Ghalfa[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 0.5j * xp.einsum(
                            "pgi, aip -> ag",
                            trial._rcholbara_chunk[iq, ik],
                            Ghalfa[:, ikpq, :, ik, :],
                            optimize=True,
                        )

                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] += 0.5 * xp.einsum(
                            "igp, aip -> ag",
                            trial._rchola_chunk[iq, ik],
                            Ghalfa[:, ik, :, ikpq, :],
                            optimize=True,
                        )
                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= 0.5 * xp.einsum(
                            "pgi, aip -> ag",
                            trial._rcholbara_chunk[iq, ik],
                            Ghalfa[:, ikpq, :, ik, :],
                            optimize=True,
                        )

                for iq in range(
                    len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)
                ):
                    iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
                    for ik in range(hamiltonian.nk):
                        ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                            0.5j
                            * xp.sqrt(2)
                            * xp.einsum(
                                "igp, aip -> ag",
                                trial._rchola_chunk[iq, ik],
                                Ghalfa[:, ik, :, ikpq, :],
                                optimize=True,
                            )
                        )
                        vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                            0.5j
                            * xp.sqrt(2)
                            * xp.einsum(
                                "pgi, aip -> ag",
                                trial._rcholbara_chunk[iq, ik],
                                Ghalfa[:, ikpq, :, ik, :],
                                optimize=True,
                            )
                        )

                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] += (
                            0.5
                            * xp.sqrt(2)
                            * xp.einsum(
                                "igp, aip -> ag",
                                trial._rchola_chunk[iq, ik],
                                Ghalfa[:, ik, :, ikpq, :],
                                optimize=True,
                            )
                        )
                        vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= (
                            0.5
                            * xp.sqrt(2)
                            * xp.einsum(
                                "pgi, aip -> ag",
                                trial._rcholbara_chunk[iq, ik],
                                Ghalfa[:, ikpq, :, ik, :],
                                optimize=True,
                            )
                        )

            receivers = handler.receivers
            for _ in range(handler.ssize - 1):
                synchronize()

                handler.scomm.Isend(Ghalfa_send, dest=receivers[srank], tag=1)
                handler.scomm.Isend(vbias_batch_plus_send, dest=receivers[srank], tag=2)
                handler.scomm.Isend(vbias_batch_minus_send, dest=receivers[srank], tag=3)

                sender = numpy.where(receivers == srank)[0]
                req1 = handler.scomm.Irecv(Ghalfa_recv, source=sender, tag=1)
                req2 = handler.scomm.Irecv(vbias_batch_plus_recv, source=sender, tag=2)
                req3 = handler.scomm.Irecv(vbias_batch_minus_recv, source=sender, tag=3)
                req1.wait()
                req2.wait()
                req3.wait()

                handler.scomm.barrier()

                # prepare sending
                vbias_batch_plus_send = vbias_batch_plus_recv.copy()
                vbias_batch_minus_send = vbias_batch_minus_recv.copy()
                if len(hamiltonian.Sset) > 0:
                    iSset = xp.arange(len(hamiltonian.Sset))
                    ik_Sset = (
                        xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Sset))
                        .reshape(hamiltonian.nk, len(hamiltonian.Sset))
                        .T
                    )
                    ikpq_S = hamiltonian.ikpq_mat[hamiltonian.Sset]
                    Gak_kpq = Ghalfa_recv[:, ik_Sset, :, ikpq_S, :]
                    Gakpq_k = Ghalfa_recv[:, ikpq_S, :, ik_Sset, :]
                    tmp1 = xp.einsum(
                        "qkigp, qkaip -> agq", trial._rchola_chunk[iSset], Gak_kpq, optimize=True
                    )
                    tmp2 = xp.einsum(
                        "qkpgi, qkaip -> agq", trial._rcholbara_chunk[iSset], Gakpq_k, optimize=True
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iSset] += 0.5j * (
                        tmp1 + tmp2
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iSset] += 0.5 * (
                        tmp1 - tmp2
                    )

                if len(hamiltonian.Qplus) > 0:
                    iQplus = xp.arange(len(hamiltonian.Qplus))
                    ik_Qplus = (
                        xp.repeat(xp.arange(hamiltonian.nk), len(hamiltonian.Qplus))
                        .reshape(hamiltonian.nk, len(hamiltonian.Qplus))
                        .T
                    )
                    ikpq_Q = hamiltonian.ikpq_mat[hamiltonian.Qplus]
                    Gak_kpq = Ghalfa_recv[:, ik_Qplus, :, ikpq_Q, :]
                    Gakpq_k = Ghalfa_recv[:, ikpq_Q, :, ik_Qplus, :]
                    iQplus_real = iQplus + len(hamiltonian.Sset)
                    tmp1 = xp.einsum(
                        "qkigp, qkaip -> agq",
                        trial._rchola_chunk[iQplus_real],
                        Gak_kpq,
                        optimize=True,
                    )
                    tmp2 = xp.einsum(
                        "qkpgi, qkaip -> agq",
                        trial._rcholbara_chunk[iQplus_real],
                        Gakpq_k,
                        optimize=True,
                    )
                    vbias_batch_plus_send[:, chol_idxs_chunk[:, None], iQplus_real] += (
                        0.5j * xp.sqrt(2) * (tmp1 + tmp2)
                    )
                    vbias_batch_minus_send[:, chol_idxs_chunk[:, None], iQplus_real] += (
                        0.5 * xp.sqrt(2) * (tmp1 - tmp2)
                    )

                else:
                    for iq in range(len(hamiltonian.Sset)):
                        iq_real = hamiltonian.Sset[iq]
                        for ik in range(hamiltonian.nk):
                            ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                            vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 0.5j * xp.einsum(
                                "igp, aip -> ag",
                                trial._rchola_chunk[iq, ik],
                                Ghalfa_recv[:, ik, :, ikpq, :],
                                optimize=True,
                            )
                            vbias_batch_plus_send[:, chol_idxs_chunk, iq] += 0.5j * xp.einsum(
                                "pgi, aip -> ag",
                                trial._rcholbara_chunk[iq, ik],
                                Ghalfa_recv[:, ikpq, :, ik, :],
                                optimize=True,
                            )

                            vbias_batch_minus_send[:, chol_idxs_chunk, iq] += 0.5 * xp.einsum(
                                "igp, aip -> ag",
                                trial._rchola_chunk[iq, ik],
                                Ghalfa_recv[:, ik, :, ikpq, :],
                                optimize=True,
                            )
                            vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= 0.5 * xp.einsum(
                                "pgi, aip -> ag",
                                trial._rcholbara_chunk[iq, ik],
                                Ghalfa_recv[:, ikpq, :, ik, :],
                                optimize=True,
                            )

                    for iq in range(
                        len(hamiltonian.Sset), len(hamiltonian.Sset) + len(hamiltonian.Qplus)
                    ):
                        iq_real = hamiltonian.Qplus[iq - len(hamiltonian.Sset)]
                        for ik in range(hamiltonian.nk):
                            ikpq = hamiltonian.ikpq_mat[iq_real, ik]
                            vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                                0.5j
                                * xp.sqrt(2)
                                * xp.einsum(
                                    "igp, aip -> ag",
                                    trial._rchola_chunk[iq, ik],
                                    Ghalfa_recv[:, ik, :, ikpq, :],
                                    optimize=True,
                                )
                            )
                            vbias_batch_plus_send[:, chol_idxs_chunk, iq] += (
                                0.5j
                                * xp.sqrt(2)
                                * xp.einsum(
                                    "pgi, aip -> ag",
                                    trial._rcholbara_chunk[iq, ik],
                                    Ghalfa_recv[:, ikpq, :, ik, :],
                                    optimize=True,
                                )
                            )

                            vbias_batch_minus_send[:, chol_idxs_chunk, iq] += (
                                0.5
                                * xp.sqrt(2)
                                * xp.einsum(
                                    "igp, aip -> ag",
                                    trial._rchola_chunk[iq, ik],
                                    Ghalfa_recv[:, ik, :, ikpq, :],
                                    optimize=True,
                                )
                            )
                            vbias_batch_minus_send[:, chol_idxs_chunk, iq] -= (
                                0.5
                                * xp.sqrt(2)
                                * xp.einsum(
                                    "pgi, aip -> ag",
                                    trial._rcholbara_chunk[iq, ik],
                                    Ghalfa_recv[:, ikpq, :, ik, :],
                                    optimize=True,
                                )
                            )
                Ghalfa_send = Ghalfa_recv.copy()

            synchronize()
            handler.scomm.Isend(vbias_batch_plus_send, dest=receivers[srank], tag=1)
            handler.scomm.Isend(vbias_batch_minus_send, dest=receivers[srank], tag=2)

            sender = numpy.where(receivers == srank)[0]
            req1 = handler.scomm.Irecv(vbias_batch_plus_recv, source=sender, tag=1)
            req2 = handler.scomm.Irecv(vbias_batch_minus_recv, source=sender, tag=2)
            req1.wait()
            req2.wait()
            handler.scomm.barrier()

            vbias_plus = vbias_batch_plus_recv.copy()
            vbias_minus = vbias_batch_minus_recv.copy()
            synchronize()
    return vbias_plus, vbias_minus
