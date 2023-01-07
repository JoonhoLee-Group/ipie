
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
# Authors: Fionn Malone <fmalone@google.com>
#          Joonho Lee <linusjoonho@gmail.com>
#

import ast
import json
import os
from typing import Tuple, Union
import sys

import h5py
import numpy
import scipy.sparse

from ipie.utils.linalg import (modified_cholesky, molecular_orbitals_rhf,
                               molecular_orbitals_uhf)
from ipie.utils.misc import merge_dicts, serialise


def write_hamiltonian(
        hcore: numpy.ndarray,
        LXmn: numpy.ndarray,
        e0: float,
        filename: str='hamiltonian.h5') -> None:
    assert len(hcore.shape) == 2, "Incorrect shape for hcore, expected 2-dimensional array"
    nmo = hcore.shape[0]
    naux = LXmn.size // (nmo*nmo)
    assert len(LXmn.shape) == 3, "Incorrect shape for LXmn, expected 3-dimensional array"
    message = f"Incorrect first dimension for LXmn: found {LXmn.shape[0]} expected {naux}"
    assert LXmn.shape[0] == naux, message
    with h5py.File(filename, 'w') as fh5:
        fh5['hcore'] = hcore
        fh5['LXmn'] = LXmn
        fh5['e0'] = e0


def read_hamiltonian(filename: str) -> Tuple[numpy.ndarray, numpy.ndarray, float]:
    with h5py.File(filename, 'r') as fh5:
        hcore = fh5['hcore'][:]
        LXmn = fh5['LXmn'][:]
        e0 = float(fh5['e0'][()])
    assert len(hcore.shape) == 2, "Incorrect shape for hcore, expected 2-dimensional array"
    nmo = hcore.shape[0]
    naux = LXmn.size // (nmo*nmo)
    assert len(LXmn.shape) == 3, "Incorrect shape for LXmn, expected 3-dimensional array"
    message = f"Incorrect first dimension for LXmn: found {LXmn.shape[0]} expected {naux}"
    assert LXmn.shape[0] == naux, message
    return hcore, LXmn, e0


def write_wavefunction(
        wfn: Union[tuple, numpy.ndarray, list],
        filename: str='wavefunction.h5',
        phi0: Union[None, list]=None
        ) -> None:
    if isinstance(wfn, numpy.ndarray) or isinstance(wfn, list):
        write_single_det_wavefunction(wfn, filename, phi0=phi0)
    else:
        if len(wfn) == 3:
            write_particle_hole_wavefunction(wfn, filename, phi0=phi0)
        elif len(wfn) == 2:
            write_noci_wavefunction(wfn, filename, phi0=phi0)
        else:
            raise RuntimeError("Unknown wavefunction time.")


def read_wavefunction(filename: str):
    try:
        return read_particle_hole_wavefunction(filename)
    except KeyError:
        pass
    try:
        return read_noci_wavefunction(filename)
    except KeyError:
        pass
    try:
        return read_single_det_wavefunction(filename)
    except:
        raise RuntimeError("Unknown file format.")


def write_single_det_wavefunction(
        wfn: Union[numpy.ndarray, list],
        filename: str,
        phi0: Union[None, list]=None
        ) -> None:
    with h5py.File(filename, 'w') as fh5:
        if isinstance(wfn, list) or len(wfn.shape) == 3:
            assert len(wfn) == 2, "Expected list for UHF wavefunction."
            fh5['psi_T_alpha'] = wfn[0]
            fh5['psi_T_beta'] = wfn[1]
            if phi0 is None:
                fh5['phi0_alpha'] = wfn[0]
                fh5['phi0_beta'] = wfn[1]
            else:
                fh5['phi0_alpha'] = phi0[0]
                fh5['phi0_beta'] = phi0[1]
        else:
            assert len(wfn.shape) == 2, "Expected 2D array for RHF wavefunction."
            fh5['psi_T_alpha'] = wfn
            if phi0 is None:
                fh5['phi0_alpha'] = wfn
            else:
                fh5['phi0_alpha'] = phi0[0]


def write_particle_hole_wavefunction(
        wfn: tuple,
        filename: str,
        phi0: Union[None, list]=None
        ) -> None:
    assert len(wfn) == 3, "Expected (ci, occa, occb)."
    with h5py.File(filename, 'w') as fh5:
        fh5['ci_coeffs'] = wfn[0]
        fh5['occ_alpha'] = wfn[1]
        fh5['occ_beta'] = wfn[2]


def write_noci_wavefunction(
        wfn: tuple,
        filename: str,
        phi0: Union[None, list]=None
        ) -> None:
    assert len(wfn) == 2, "Expected (ci, psi)."
    assert isinstance(wfn[1], list)
    assert len(wfn[1][0].shape) == 3, "Expected psi.shape = (ndet, nmo, nocca)"
    assert len(wfn[1][1].shape) == 3, "Expected psi.shape = (ndet, nmo, noccb)"
    ndet = len(wfn[0])
    with h5py.File(filename, 'w') as fh5:
        fh5['ci_coeffs'] = wfn[0]
        fh5['psi_T_alpha'] = wfn[1][0]
        fh5['psi_T_beta'] = wfn[1][1]
        if phi0 is None:
            fh5['phi0_alpha'] = wfn[1][0][0]
            fh5['phi0_beta'] = wfn[1][1][0]
        else:
            fh5['phi0_alpha'] = phi0[0]
            fh5['phi0_beta'] = phi0[1]


def read_particle_hole_wavefunction(
        filename: str
        ) -> Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
    with h5py.File(filename, 'r') as fh5:
        ci_coeffs = fh5['ci_coeffs'][:]
        occ_alpha = fh5['occ_alpha'][:]
        occ_beta = fh5['occ_beta'][:]
    return (ci_coeffs, occ_alpha, occ_beta), None


def read_noci_wavefunction(
        filename: str
        ) -> Tuple[numpy.ndarray, list]:
    with h5py.File(filename, 'r') as fh5:
        ci_coeffs = fh5['ci_coeffs'][:]
        ndets = len(ci_coeffs)
        for idet in range(ndets):
            psia = fh5[f'psi_T_alpha'][:]
            psib = fh5[f'psi_T_beta'][:]
    return (ci_coeffs, [psia, psib]), None


def read_single_det_wavefunction(
        filename: str
        ) -> Tuple[numpy.ndarray, list]:
    with h5py.File(filename, 'r') as fh5:
        psia = fh5['psi_T_alpha'][:]
        phi0a = fh5['phi0_alpha'][:]
        try:
            psib = fh5['psi_T_beta'][:]
            phi0b = fh5['phi0_beta'][:]
            wfn = [psia, psib]
            phi0 = [phi0a, phi0b]
        except KeyError:
            wfn = [psia, psia]
            phi0 = [phi0a, phi0a]
    return wfn, phi0


def format_fixed_width_strings(strings):
    return " ".join("{:>23}".format(s) for s in strings)


def format_fixed_width_floats(floats):
    return " ".join("{: .16e}".format(f) for f in floats)

def format_fixed_width_cmplx(floats):
    return " ".join("{: .10e} {: .10e}".format(f.real, f.imag) for f in floats)

def write_json_input_file(
        input_filename: str,
        hamil_filename: str,
        wfn_filename: str,
        nelec: tuple,
        num_walkers: int=640,
        timestep: float=0.005,
        num_blocks: float=10,
        estimates_filename: str='estimates.0.h5',
        options: dict={},
        ):
    na, nb = nelec
    basic = {
        "system": {
            "nup": na,
            "ndown": nb,
        },
        "hamiltonian": {"name": "Generic", "integrals": hamil_filename},
        "qmc": {
            "dt": timestep,
            "nwalkers": num_walkers,
            "nsteps": 25,
            "blocks": num_blocks,
            "batched": True,
            "pop_control_freq": 5,
            "stabilise_freq": 5,
        },
        "trial": {"filename": wfn_filename},
        "estimators": {"filename": estimates_filename},
    }
    full = merge_dicts(basic, options)
    with open(input_filename, "w") as f:
        f.write(json.dumps(full, indent=4, separators=(",", ": ")))


def to_json(afqmc):
    json.encoder.FLOAT_REPR = lambda o: format(o, ".6f")
    json_string = json.dumps(
        serialise(afqmc, verbose=afqmc.verbosity), sort_keys=False, indent=4
    )
    return json_string


def get_input_value(inputs, key, default=0, alias=None, verbose=False):
    """Helper routine to parse input options."""
    val = inputs.get(key, None)
    if val is not None and verbose:
        if isinstance(val, dict):
            print("# Options for {}".format(key))
            for k, v in val.items():
                print("# Setting {} to {}.".format(k, v))
        else:
            print("# Setting {} to {}.".format(key, val))
    if val is None:
        if alias is not None:
            for a in alias:
                val = inputs.get(a, None)
                if val is not None:
                    if verbose:
                        print("# Setting {} to {}.".format(key, val))
                    break
        if val is None:
            val = default
            if verbose:
                print(
                    "# Note: {} not specified. Setting to default value"
                    " of {}.".format(key, default)
                )
    return val


def read_qmcpack_wfn_hdf(filename, nelec=None):
    try:
        with h5py.File(filename, "r") as fh5:
            wgroup = fh5["Wavefunction/NOMSD"]
            wfn, psi0 = read_qmcpack_nomsd_hdf5(wgroup, nelec=nelec)
    except KeyError:
        with h5py.File(filename, "r") as fh5:
            wgroup = fh5["Wavefunction/PHMSD"]
            wfn, psi0 = read_qmcpack_phmsd_hdf5(wgroup, nelec=nelec)
    except KeyError:
        print("Wavefunction not found.")
        sys.exit()
    return wfn, psi0


def read_qmcpack_nomsd_hdf5(wgroup, nelec=None):
    dims = wgroup["dims"]
    nmo = dims[0]
    na = dims[1]
    nb = dims[2]
    if nelec is not None:
        log = "Number of electrons does not match wavefunction: {} vs {}."
        assert na == nelec[0], log.format(na, nelec[0])
        assert nb == nelec[0], log.format(nb, nelec[1])
    walker_type = dims[3]
    if walker_type == 2:
        uhf = True
    else:
        uhf = False
    nci = dims[4]
    coeffs = from_qmcpack_complex(wgroup["ci_coeffs"][:], (nci,))
    psi0a = from_qmcpack_complex(wgroup["Psi0_alpha"][:], (nmo, na))
    if uhf:
        psi0b = from_qmcpack_complex(wgroup["Psi0_beta"][:], (nmo, nb))
    psi0 = numpy.zeros((nmo, na + nb), dtype=numpy.complex128)
    psi0[:, :na] = psi0a.copy()
    if uhf:
        psi0[:, na:] = psi0b.copy()
    else:
        psi0[:, na:] = psi0a[:, :nb].copy()
    wfn = numpy.zeros((nci, nmo, na + nb), dtype=numpy.complex128)
    for idet in range(nci):
        ix = 2 * idet if uhf else idet
        pa = orbs_from_dset(wgroup["PsiT_{:d}/".format(idet)])
        wfn[idet, :, :na] = pa
        if uhf:
            ix = 2 * idet + 1
            wfn[idet, :, na:] = orbs_from_dset(wgroup["PsiT_{:d}/".format(ix)])
        else:
            wfn[idet, :, na:] = pa[:, :nb]
    return (coeffs, wfn), psi0


def read_qmcpack_phmsd_hdf5(wgroup, nelec=None):
    dims = wgroup["dims"]
    nmo = dims[0]
    na = dims[1]
    nb = dims[2]
    if nelec is not None:
        log = "Number of electrons does not match wavefunction: {} vs {}."
        assert na == nelec[0], log.format(na, nelec[0])
        assert nb == nelec[0], log.format(nb, nelec[1])
    walker_type = dims[3]
    if walker_type == 2:
        uhf = True
    else:
        uhf = False
    nci = dims[4]
    coeffs = from_qmcpack_complex(wgroup["ci_coeffs"][:], (nci,))
    occs = wgroup["occs"][:].reshape((nci, na + nb))
    occa = occs[:, :na]
    occb = occs[:, na:] - nmo
    wfn = (coeffs, occa, occb)
    psi0a = from_qmcpack_complex(wgroup["Psi0_alpha"][:], (nmo, na))
    if uhf:
        psi0b = from_qmcpack_complex(wgroup["Psi0_beta"][:], (nmo, nb))
    psi0 = numpy.zeros((nmo, na + nb), dtype=numpy.complex128)
    psi0[:, :na] = psi0a.copy()
    if uhf:
        psi0[:, na:] = psi0b.copy()
    else:
        psi0[:, na:] = psi0a.copy()
    return wfn, psi0


def write_qmcpack_wfn(filename, wfn, walker_type, nelec, norb, init=None, mode="w"):
    # User defined wavefunction.
    # PHMSD is a list of tuple of (ci, occa, occb).
    # NOMSD is a tuple of (list, numpy.ndarray).
    if len(wfn) == 3:
        coeffs, occa, occb = wfn
        wfn_type = "PHMSD"
    elif len(wfn) == 2:
        coeffs, wfn = wfn
        wfn_type = "NOMSD"
    else:
        print("Unknown wavefunction type passed.")
        sys.exit()

    with h5py.File(filename, mode) as fh5:
        nalpha, nbeta = nelec
        # TODO: FIX for GHF eventually.
        if walker_type == "ghf":
            walker_type = 3
        elif walker_type == "uhf":
            walker_type = 2
            uhf = True
        else:
            walker_type = 1
            uhf = False
        if wfn_type == "PHMSD":
            walker_type = 2
        if wfn_type == "NOMSD":
            try:
                wfn_group = fh5.create_group("Wavefunction/NOMSD")
            except ValueError:
                del fh5["Wavefunction/NOMSD"]
                wfn_group = fh5.create_group("Wavefunction/NOMSD")
            write_nomsd(wfn_group, wfn, uhf, nelec, init=init)
        else:
            try:
                wfn_group = fh5.create_group("Wavefunction/PHMSD")
            except ValueError:
                # print(" # Warning: Found existing wavefunction group. Removing.")
                del fh5["Wavefunction/PHMSD"]
                wfn_group = fh5.create_group("Wavefunction/PHMSD")
            write_phmsd(wfn_group, occa, occb, nelec, norb, init=init)
        wfn_group["ci_coeffs"] = to_qmcpack_complex(coeffs)
        dims = [norb, nalpha, nbeta, walker_type, len(coeffs)]
        wfn_group["dims"] = numpy.array(dims, dtype=numpy.int32)


def write_nomsd(fh5, wfn, uhf, nelec, thresh=1e-8, init=None):
    """Write NOMSD to HDF.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    wfn : :class:`numpy.ndarray`
        NOMSD trial wavefunctions.
    uhf : bool
        UHF style wavefunction.
    nelec : tuple
        Number of alpha and beta electrons.
    thresh : float
        Threshold for writing wavefunction elements.
    """
    nalpha, nbeta = nelec
    wfn[abs(wfn) < thresh] = 0.0
    if len(wfn.shape) == 2:
        nmo = wfn.shape[0]
        nel = wfn.shape[1]
        wfn = wfn.reshape((1, nmo, nel))
    if init is not None:
        fh5["Psi0_alpha"] = to_qmcpack_complex(init[0])
        fh5["Psi0_beta"] = to_qmcpack_complex(init[1])
    else:
        fh5["Psi0_alpha"] = to_qmcpack_complex(
            numpy.array(wfn[0, :, :nalpha].copy(), dtype=numpy.complex128)
        )
        if uhf:
            fh5["Psi0_beta"] = to_qmcpack_complex(
                numpy.array(wfn[0, :, nalpha:].copy(), dtype=numpy.complex128)
            )
    for idet, w in enumerate(wfn):
        # QMCPACK stores this internally as a csr matrix, so first convert.
        ix = 2 * idet if uhf else idet
        psia = scipy.sparse.csr_matrix(w[:, :nalpha].conj().T)
        write_nomsd_single(fh5, psia, ix)
        if uhf:
            ix = 2 * idet + 1
            psib = scipy.sparse.csr_matrix(w[:, nalpha:].conj().T)
            write_nomsd_single(fh5, psib, ix)


def write_nomsd_single(fh5, psi, idet):
    """Write single component of NOMSD to hdf.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    psi : :class:`scipy.sparse.csr_matrix`
        Sparse representation of trial wavefunction.
    idet : int
        Determinant number.
    """
    base = "PsiT_{:d}/".format(idet)
    dims = [psi.shape[0], psi.shape[1], psi.nnz]
    fh5[base + "dims"] = numpy.array(dims, dtype=numpy.int32)
    fh5[base + "data_"] = to_qmcpack_complex(psi.data)
    fh5[base + "jdata_"] = psi.indices
    fh5[base + "pointers_begin_"] = psi.indptr[:-1]
    fh5[base + "pointers_end_"] = psi.indptr[1:]


def write_phmsd(fh5, occa, occb, nelec, norb, init=None):
    """Write NOMSD to HDF.

    Parameters
    ----------
    fh5 : h5py group
        Wavefunction group to write to file.
    nelec : tuple
        Number of alpha and beta electrons.
    """
    # TODO: Update if we ever wanted "mixed" phmsd type wavefunctions.
    na, nb = nelec
    if init is not None:
        psi0 = numpy.array(init[0], numpy.complex128)
        fh5["Psi0_alpha"] = to_qmcpack_complex(psi0)
        psi0 = numpy.array(init[1], numpy.complex128)
        fh5["Psi0_beta"] = to_qmcpack_complex(psi0)
    else:
        init = numpy.eye(norb, dtype=numpy.complex128)
        fh5["Psi0_alpha"] = to_qmcpack_complex(init[:, occa[0]].copy())
        fh5["Psi0_beta"] = to_qmcpack_complex(init[:, occb[0]].copy())
    fh5["fullmo"] = numpy.array([0], dtype=numpy.int32)
    fh5["type"] = 0
    occs = numpy.zeros((len(occa), na + nb), dtype=numpy.int32)
    occs[:, :na] = numpy.array(occa)
    occs[:, na:] = norb + numpy.array(occb)
    # Reading 1D array currently in qmcpack.
    fh5["occs"] = occs.ravel()


def orbs_from_dset(dset):
    """Will read actually A^{H} but return A."""
    dims = dset["dims"][:]
    wfn_shape = (dims[0], dims[1])
    nnz = dims[2]
    data = from_qmcpack_complex(dset["data_"][:], (nnz,))
    indices = dset["jdata_"][:]
    pbb = dset["pointers_begin_"][:]
    pbe = dset["pointers_end_"][:]
    indptr = numpy.zeros(dims[0] + 1)
    indptr[:-1] = pbb
    indptr[-1] = pbe[-1]
    wfn = scipy.sparse.csr_matrix((data, indices, indptr), shape=wfn_shape)
    return wfn.toarray().conj().T.copy()


def to_qmcpack_complex(array):
    shape = array.shape
    return array.view(numpy.float64).reshape(shape + (2,))

def from_qmcpack_dense(filename):
    with h5py.File(filename, "r") as fh5:
        enuc = fh5["Hamiltonian/Energies"][:][0]
        dims = fh5["Hamiltonian/dims"][:]
        nmo = dims[3]
        nchol = dims[-1]
        real_ints = False
        try:
            hcore = fh5["Hamiltonian/hcore"][:]
            hcore = from_qmcpack_complex(hcore, (nmo, nmo))
            chol = fh5["Hamiltonian/DenseFactorized/L"][:]
            chol = from_qmcpack_complex(chol, (nmo * nmo, -1))
        except ValueError:
            # Real format.
            hcore = fh5["Hamiltonian/hcore"][:]
            chol = fh5["Hamiltonian/DenseFactorized/L"][:]
            real_ints = True
        nalpha = dims[4]
        nbeta = dims[5]
        return (hcore, chol, enuc, int(nmo), int(nalpha), int(nbeta))

def from_qmcpack_sparse(filename):
    with h5py.File(filename, "r") as fh5:
        enuc = fh5["Hamiltonian/Energies"][:][0]
        dims = fh5["Hamiltonian/dims"][:]
        nmo = dims[3]
        real_ints = False
        try:
            hcore = fh5["Hamiltonian/hcore"][:]
            hcore = hcore.view(numpy.complex128).reshape(nmo, nmo)
        except KeyError:
            # Old sparse format.
            hcore = fh5["Hamiltonian/H1"][:].view(numpy.complex128).ravel()
            idx = fh5["Hamiltonian/H1_indx"][:]
            row_ix = idx[::2]
            col_ix = idx[1::2]
            hcore = scipy.sparse.csr_matrix((hcore, (row_ix, col_ix))).toarray()
            hcore = numpy.tril(hcore, -1) + numpy.tril(hcore, 0).conj().T
        except ValueError:
            # Real format.
            hcore = fh5["Hamiltonian/hcore"][:]
            real_ints = True
        chunks = dims[2]
        block_sizes = fh5["Hamiltonian/Factorized/block_sizes"][:]
        nchol = dims[7]
        nval = sum(block_sizes)
        if real_ints:
            vals = numpy.zeros(nval, dtype=numpy.float64)
        else:
            vals = numpy.zeros(nval, dtype=numpy.complex128)
        row_ix = numpy.zeros(nval, dtype=numpy.int32)
        col_ix = numpy.zeros(nval, dtype=numpy.int32)
        s = 0
        for ic, bs in enumerate(block_sizes):
            ixs = fh5["Hamiltonian/Factorized/index_%i" % ic][:]
            row_ix[s : s + bs] = ixs[::2]
            col_ix[s : s + bs] = ixs[1::2]
            if real_ints:
                vals[s : s + bs] = numpy.real(
                    fh5["Hamiltonian/Factorized/vals_%i" % ic][:]
                ).ravel()
            else:
                vals[s : s + bs] = (
                    fh5["Hamiltonian/Factorized/vals_%i" % ic][:]
                    .view(numpy.complex128)
                    .ravel()
                )
            s += bs
        nalpha = dims[4]
        nbeta = dims[5]
        chol_vecs = scipy.sparse.csr_matrix(
            (vals, (row_ix, col_ix)), shape=(nmo * nmo, nchol)
        )
        return (hcore, chol_vecs, enuc, int(nmo), int(nalpha), int(nbeta))

def write_qmcpack_dense(
    hcore,
    chol,
    nelec,
    nmo,
    enuc=0.0,
    filename="hamiltonian.h5",
    real_chol=True,
    verbose=False,
    ortho=None,
):
    assert len(chol.shape) == 2
    assert chol.shape[0] == nmo * nmo
    with h5py.File(filename, "w") as fh5:
        fh5["Hamiltonian/Energies"] = numpy.array([enuc, 0])
        if real_chol:
            fh5["Hamiltonian/hcore"] = numpy.real(hcore)
            fh5["Hamiltonian/DenseFactorized/L"] = numpy.real(chol)
        else:
            fh5["Hamiltonian/hcore"] = to_qmcpack_complex(
                hcore.astype(numpy.complex128)
            )
            fh5["Hamiltonian/DenseFactorized/L"] = to_qmcpack_complex(
                chol.astype(numpy.complex128)
            )
        fh5["Hamiltonian/dims"] = numpy.array(
            [0, 0, 0, nmo, nelec[0], nelec[1], 0, chol.shape[-1]]
        )
        if ortho is not None:
            fh5["Hamiltonian/X"] = ortho

def write_qmcpack_sparse(
    hcore,
    chol,
    nelec,
    nmo,
    enuc=0.0,
    filename="hamiltonian.h5",
    real_chol=False,
    verbose=False,
    cutoff=1e-16,
    ortho=None,
):
    with h5py.File(filename, "w") as fh5:
        fh5["Hamiltonian/Energies"] = numpy.array([enuc, 0])
        if real_chol:
            fh5["Hamiltonian/hcore"] = hcore
        else:
            shape = hcore.shape
            hcore = hcore.astype(numpy.complex128).view(numpy.float64)
            hcore = hcore.reshape(shape + (2,))
            fh5["Hamiltonian/hcore"] = hcore
        if ortho is not None:
            fh5["Hamiltonian/X"] = ortho
        # number of cholesky vectors
        nchol_vecs = chol.shape[-1]
        ix, vals = to_sparse(chol, cutoff=cutoff)
        nnz = len(vals)
        mem = (8 if real_chol else 16) * nnz / (1024.0**3)
        if verbose:
            print(
                " # Total number of non-zero elements in sparse cholesky ERI"
                " tensor: %d" % nnz
            )
            nelem = chol.shape[0] * chol.shape[1]
            print(
                " # Sparsity of ERI Cholesky tensor: " "%f" % (1 - float(nnz) / nelem)
            )
            print(" # Total memory required for ERI tensor: %13.8e GB" % (mem))
        fh5["Hamiltonian/Factorized/block_sizes"] = numpy.array([nnz])
        fh5["Hamiltonian/Factorized/index_0"] = numpy.array(ix)
        if real_chol:
            fh5["Hamiltonian/Factorized/vals_0"] = numpy.array(vals)
        else:
            fh5["Hamiltonian/Factorized/vals_0"] = to_qmcpack_complex(
                numpy.array(vals, dtype=numpy.complex128)
            )
        # Number of integral blocks used for chunked HDF5 storage.
        # Currently hardcoded for simplicity.
        nint_block = 1
        (nalpha, nbeta) = nelec
        unused = 0
        fh5["Hamiltonian/dims"] = numpy.array(
            [unused, nnz, nint_block, nmo, nalpha, nbeta, unused, nchol_vecs]
        )
        occups = [i for i in range(0, nalpha)]
        occups += [i + nmo for i in range(0, nbeta)]
        fh5["Hamiltonian/occups"] = numpy.array(occups)


def read_fortran_complex_numbers(filename):
    with open(filename) as f:
        content = f.readlines()
    # Converting fortran complex numbers to python. ugh
    # Be verbose for clarity.
    useable = [c.strip() for c in content]
    tuples = [ast.literal_eval(u) for u in useable]
    orbs = [complex(t[0], t[1]) for t in tuples]
    return numpy.array(orbs)

def fcidump_header(nel, norb, spin):
    header = (
        "&FCI\n"
        + "NORB=%d,\n" % int(norb)
        + "NELEC=%d,\n" % int(nel)
        + "MS2=%d,\n" % int(spin)
        + "UHF=.FALSE.,\n"
        + "ORBSYM="
        + ",".join([str(1)] * norb)
        + ",\n"
        "&END\n"
    )
    return header


def read_qmcpack_wfn(filename, skip=9):
    with open(filename) as f:
        content = f.readlines()[skip:]
    useable = numpy.array([c.split() for c in content]).flatten()
    tuples = [ast.literal_eval(u) for u in useable]
    orbs = [complex(t[0], t[1]) for t in tuples]
    return numpy.array(orbs)


def from_qmcpack_complex(data, shape):
    return data.view(numpy.complex128).ravel().reshape(shape)

def to_sparse(vals, offset=0, cutoff=1e-8):
    nz = numpy.where(numpy.abs(vals) > cutoff)
    ix = numpy.empty(nz[0].size + nz[1].size, dtype=numpy.int32)
    ix[0::2] = nz[0]
    ix[1::2] = nz[1]
    vals = numpy.array(vals[nz], dtype=numpy.complex128)
    return ix, vals
