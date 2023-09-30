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
#          Joonho Lee
#

"""Generate AFQMC data from PYSCF (molecular) simulation."""
import math
import time
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import h5py
import numpy as np
import scipy.linalg

# pylint: disable=import-error
from pyscf import gto, lib, scf

from ipie.estimators.generic import core_contribution_cholesky
from ipie.hamiltonians.generic import GenericRealChol
from ipie.legacy.estimators.generic import local_energy_generic_cholesky
from ipie.legacy.estimators.greens_function import gab
from ipie.utils.io import write_hamiltonian, write_wavefunction
from ipie.utils.misc import dotdict


def gen_ipie_input_from_pyscf_chk(
    pyscf_chkfile: str,
    hamil_file: str = "hamiltonian.h5",
    wfn_file: str = "wavefunction.h5",
    verbose: bool = True,
    chol_cut: float = 1e-5,
    ortho_ao: bool = False,
    mcscf: bool = False,
    linear_dep_thresh: float = 1e-8,
    num_frozen_core: int = 0,
) -> None:
    if mcscf:
        scf_data = load_from_pyscf_chkfile(pyscf_chkfile, base="mcscf")
    else:
        scf_data = load_from_pyscf_chkfile(pyscf_chkfile)
    mol = scf_data["mol"]
    hcore = scf_data["hcore"]
    ortho_ao_mat = scf_data["X"]
    mo_coeffs = scf_data["mo_coeff"]
    mo_occ = scf_data["mo_occ"]
    if ortho_ao:
        basis_change_matrix = ortho_ao_mat
    else:
        basis_change_matrix = mo_coeffs

        if isinstance(mo_coeffs, list) or len(mo_coeffs.shape) == 3:
            if verbose:
                print(
                    "# UHF mo coefficients found and ortho-ao == False. Using"
                    " alpha mo coefficients for basis transformation."
                )
            basis_change_matrix = mo_coeffs[0]
    ham = generate_hamiltonian(
        mol,
        mo_coeffs,
        hcore,
        basis_change_matrix,
        chol_cut=chol_cut,
        num_frozen_core=num_frozen_core,
        verbose=verbose,
    )
    write_hamiltonian(ham.H1[0], copy_LPX_to_LXmn(ham.chol), ham.ecore, filename=hamil_file)
    nelec = (mol.nelec[0] - num_frozen_core, mol.nelec[1] - num_frozen_core)
    if verbose:
        print(f"# Number of electrons in simulation: {nelec}")
    if mcscf:
        ci_coeffs = scf_data["ci_coeffs"]
        occa = scf_data["occa"]
        occb = scf_data["occb"]
        write_wavefunction((ci_coeffs, occa, occb), wfn_file)
    else:
        wfn = generate_wavefunction_from_mo_coeff(
            mo_coeffs,
            mo_occ,
            basis_change_matrix,
            nelec,
            ortho_ao=ortho_ao,
            num_frozen_core=num_frozen_core,
        )
        write_wavefunction(wfn, wfn_file)


@dataclass(frozen=True)
class Wavefunction:
    wfn: Union[Tuple, np.ndarray]


def copy_LXmn_to_LPX(chol: np.ndarray) -> np.ndarray:
    assert len(chol.shape) == 3
    assert chol.shape[-1] == chol.shape[-2]
    naux, nbasis, _ = chol.shape
    return chol.reshape(naux, nbasis * nbasis).T.copy()


def copy_LPX_to_LXmn(chol: np.ndarray) -> np.ndarray:
    assert len(chol.shape) == 2
    nbasis_sq, naux = chol.shape
    nbasis = math.isqrt(nbasis_sq)
    return chol.reshape(nbasis, nbasis, naux).transpose((2, 0, 1)).copy()


def generate_hamiltonian_from_chk(
    pyscf_chk: str,
    use_mcscf: bool = False,
    chol_cut: float = 1e-5,
    num_frozen_core: int = 0,
    verbose: bool = False,
) -> GenericRealChol:
    """Generate ipie Hamiltonian from pyscf chkfile.

    Wrapper around generate_hamiltonian.

    Parameters
    ----------
        pyscf_chk : pyscf scf or mcscf checkpoint file.
        use_mcscf : Use MCSCF mo coefficients or not.
        chol_cut : Cholesky threshold.
        num_frozen_core : Number of core orbitals to freeze.
        verbose : print out some information.

    Returns
    -------
        ham : ipie Hamiltonian.
    """
    scf_data = load_from_pyscf_chkfile(pyscf_chk, base="mcscf" if use_mcscf else "scf")
    ham = generate_hamiltonian(
        scf_data["mol"],
        scf_data["mo_coeff"],
        scf_data["hcore"],
        scf_data["mo_coeff"],
        chol_cut=chol_cut,
        num_frozen_core=num_frozen_core,
        verbose=verbose,
    )
    return ham


def generate_hamiltonian(
    mol,
    mo_coeffs: np.ndarray,
    hcore: np.ndarray,
    basis_change_matrix: np.ndarray,
    chol_cut: float = 1e-8,
    num_frozen_core: int = 0,
    ortho_ao: bool = False,
    verbose: bool = False,
) -> GenericRealChol:
    h1e, chol, e0 = generate_integrals(
        mol, hcore, basis_change_matrix, chol_cut=chol_cut, verbose=verbose
    )
    if num_frozen_core > 0:
        assert not ortho_ao, "--ortho-ao and --frozen-core not supported together."
        assert num_frozen_core <= mol.nelec[0], f"{num_frozen_core} < {mol.nelec[0]}"
        assert num_frozen_core <= mol.nelec[1], f"{num_frozen_core} < {mol.nelec[1]}"
        h1e_eff, chol_act, e0_eff = freeze_core(
            h1e, chol, e0, mo_coeffs, num_frozen_core, verbose=verbose
        )
        return GenericRealChol(h1e_eff, copy_LXmn_to_LPX(chol_act), e0_eff)
    else:
        return GenericRealChol(np.array([h1e, h1e]), copy_LXmn_to_LPX(chol), e0)


def generate_wavefunction_from_mo_coeff(
    mo_coeff: Union[list, np.ndarray],
    mo_occ: Union[list, np.ndarray],
    X: np.ndarray,
    nelec: tuple,
    ortho_ao: bool = False,
    num_frozen_core: int = 0,
) -> Union[list, np.ndarray]:
    """Generate QMCPACK trial wavefunction."""
    uhf = isinstance(mo_coeff, list) or len(mo_coeff.shape) == 3
    if not uhf and nelec[0] != nelec[1]:
        rohf = True
    else:
        rohf = False
    nmo = X.shape[1] - num_frozen_core
    if ortho_ao:
        Xinv = scipy.linalg.inv(X)
        if uhf:
            # We are assuming C matrix is energy ordered.
            wfna = np.dot(Xinv, mo_coeff[0])[:, mo_occ[0] > 0]
            wfnb = np.dot(Xinv, mo_coeff[1])[:, mo_occ[1] > 0]
            wfn = [wfna, wfnb]
        elif rohf:
            _occ_a = mo_occ > 0
            _occ_b = mo_occ > 1
            wfna = np.dot(Xinv, mo_coeff)[:, _occ_a]
            wfnb = np.dot(Xinv, mo_coeff)[:, _occ_b]
            wfn = [wfna, wfnb]
        else:
            mo_occ = np.array(mo_occ, dtype=np.int64)
            _occ_a = mo_occ > 0
            wfna = np.dot(Xinv, mo_coeff)[:, _occ_a]
            wfn = wfna
    else:
        if uhf:
            # HP: Assuming we are working in the alpha orbital basis, and write the beta orbitals as LCAO of alpha orbitals
            I = np.identity(nmo, dtype=np.float64)
            wfna = I[:, mo_occ[0][num_frozen_core:] > 0]
            Xinv = scipy.linalg.pinv(X[:, num_frozen_core:])
            wfnb = np.dot(Xinv, mo_coeff[1])[:, num_frozen_core:]
            wfnb = wfnb[:, mo_occ[1][num_frozen_core:] > 0]
            wfn = [wfna, wfnb]
        elif rohf:
            I = np.identity(nmo, dtype=np.float64)
            _occ_a = mo_occ > 0
            _occ_b = mo_occ > 1
            wfna = I[:, _occ_a[num_frozen_core:]].copy()
            wfnb = I[:, _occ_b[num_frozen_core:]].copy()
            wfn = [wfna, wfnb]
        else:
            # Assuming we are working in MO basis, only works for RHF, ROHF trials.
            I = np.identity(nmo, dtype=np.float64)
            wfna = I[:, mo_occ[num_frozen_core:] > 0]
            wfn = wfna
    return wfn


def generate_integrals(
    mol: gto.Mole,
    hcore: np.ndarray,
    X: np.ndarray,
    chol_cut: float = 1e-5,
    verbose: bool = False,
    cas: Optional[Tuple[int, int]] = None,
):
    """Generate integrals from pyscf mol object.

    Parameters
    ----------
        mol : pyscf.gto.Mole
            pyscf molecule
        hcore : np.array
            core hamiltonian
        X : np.array
            basis rotation matrix
        chol_cut : float
            Cholesky cutoff
        verbose : bool
            Print some information during decomposition.
        cas : Tuple[int, int]
            Active space to specify. Optional. Deafault None.

    Returns
    -------
        h1e : np.ndarray
            One-body hamiltonian in MO basis, or basis defined by X.
        chol_vecs : np.ndarray
            Cholesky vectors. Output shape = (naux, nbasis, nbasis)
        enuc : float
            Core energy.
    """
    if len(X.shape) == 2:
        h1e = np.dot(X.T, np.dot(hcore, X))
    elif len(X.shape) == 3:
        h1e = np.dot(X[0].T, np.dot(hcore, X[0]))
    nbasis = h1e.shape[-1]
    # Step 2. Genrate Cholesky decomposed ERIs in non-orthogonal AO basis.
    if verbose:
        print(" # Performing modified Cholesky decomposition on ERI tensor.")
    chol_vecs = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)
    if verbose:
        print(" # Orthogonalising Cholesky vectors.")
    start = time.time()
    # Step 2.a Orthogonalise Cholesky vectors.
    if len(X.shape) == 2:
        ao2mo_chol(chol_vecs, X)
    elif len(X.shape) == 3:
        ao2mo_chol(chol_vecs, X[0])
    if verbose:
        print(f" # Time to orthogonalise: {time.time() - start:f}")
    enuc = mol.energy_nuc()

    return h1e, chol_vecs.reshape((-1, nbasis, nbasis)), enuc


def ao2mo_chol(eri, C, verbose=False):
    nb = C.shape[-1]
    for i, cv in enumerate(eri):
        if verbose and i % 100 == 0:
            print(f" # ao2mo cholesky % complete = {100 * float(i) / len(eri)} %")
        half = np.dot(cv.reshape(nb, nb), C)
        eri[i] = np.dot(C.conj().T, half).ravel()


def cholesky(
    mol, filename="hamil.h5", max_error=1e-6, verbose=False, cmax=20, CHUNK_SIZE=2.0, MAX_SIZE=20.0
):
    nao = mol.nao_nr()
    if nao * nao * cmax * nao * 8.0 / 1024.0**3 > MAX_SIZE:
        if verbose:
            print(f"# Approximate memory for cholesky > MAX_SIZE ({MAX_SIZE} GB).")
            print("# Using out of core algorithm.")
            return chunked_cholesky_outcore(
                mol,
                filename=filename,
                max_error=max_error,
                verbose=verbose,
                cmax=cmax,
                CHUNK_SIZE=CHUNK_SIZE,
            )
        else:
            return chunked_cholesky(mol, max_error=max_error, verbose=verbose, cmax=cmax)


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`np.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`np.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    # This shape is more convenient for ipie.
    chol_vecs = np.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, _, _, _ = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs." % nchol_max)
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor("int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = np.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
        chol_vecs[nchol + 1] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print("# iteration %5d: delta_max = %13.8e: time = %13.8e" % info)

    return chol_vecs[:nchol]


def chunked_cholesky_outcore(
    mol, filename="hamil.h5", max_error=1e-6, verbose=False, cmax=20, CHUNK_SIZE=2.0
):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`np.ndarray`
        Orthogonalising matrix for AOs. (e.g., mo_coeff).
    delta : float
        Accuracy desired.
    verbose : bool
        If true print out convergence progress.
    cmax : int
        nchol = cmax * M, where M is the number of basis functions.
        Controls buffer size for cholesky vectors.

    Returns
    -------
    chol_vecs : :class:`np.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = np.zeros(nao * nao)
    nchol_max = cmax * nao
    mem = 8.0 * nchol_max * nao * nao / 1024.0**3
    chunk_size = min(int(CHUNK_SIZE * 1024.0**3 / (8 * nao * nao)), nchol_max)
    if verbose:
        print(f"# Number of AOs: {nao}")
        print(f"# Writing AO Cholesky to {filename:s}.")
        print(f"# Max number of Cholesky vectors: {nchol_max}")
        print(f"# Max memory required for Cholesky tensor: {mem} GB")
        print(
            "# Splitting calculation into chunks of size: {} / GB".format(
                chunk_size, 8 * chunk_size * nao * nao / (1024.0**3)
            )
        )
        print("# Generating diagonal.")
    chol_vecs = np.zeros((chunk_size, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    start = time.time()
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, _, _, _ = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = np.argmax(diag)
    delta_max = diag[nu]
    with h5py.File(filename, "w") as fh5:
        fh5.create_dataset("Lao", shape=(nchol_max, nao * nao), dtype=np.float64)
    end = time.time()
    if verbose:
        print(f"# Time to generate diagonal {end - start} s.")
        print("# Generating Cholesky decomposition of ERIs.")
        print(f"# iteration {0:5d}: delta_max = {delta_max:13.8e}")
    j = nu // nao
    l = nu % nao
    sj = np.searchsorted(dims, j)
    sl = np.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = np.zeros(nao * nao)
    eri_col = mol.intor("int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1))
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = np.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5

    def compute_residual(chol, ichol, nchol, nu):
        # Updated residual = \sum_x L_i^x L_nu^x
        # R = np.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        R = 0.0
        with h5py.File(filename, "r") as fh5:
            for ic in range(0, ichol):
                # Compute dot product from file.
                # print(ichol*chunk_size, (ichol+1)*chunk_size)
                # import sys
                # sys.exit()
                # print(ic*chunk_size, (ic*chunk_size)
                L = fh5["Lao"][ic * chunk_size : (ic + 1) * chunk_size, :]
                R += np.dot(L[:, nu], L[:, :])
        R += np.dot(chol[: nchol + 1, nu], chol[: nchol + 1, :])
        return R

    nchol = 0
    ichunk = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol % chunk_size] * chol_vecs[nchol % chunk_size]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = np.argmax(np.abs(delta))
        delta_max = np.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = np.searchsorted(dims, j)
        sl = np.searchsorted(dims, l)
        if dims[sj] != j and j != 0:
            sj -= 1
        if dims[sl] != l and l != 0:
            sl -= 1
        # Compute ERI chunk.
        eri_col = mol.intor(
            "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
        )
        # Select correct ERI chunk from shell.
        cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
        Munu0 = eri_col[:, :, cj, cl].reshape(nao * nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        startr = time.time()
        R = compute_residual(chol_vecs, ichunk, nchol % chunk_size, nu)
        endr = time.time()
        if nchol > 0 and (nchol + 1) % chunk_size == 0:
            startw = time.time()
            # delta = L[ichunk*chunk_size:(ichunk+1)*chunk_size]-chol_vecs
            with h5py.File(filename, "r+") as fh5:
                fh5["Lao"][ichunk * chunk_size : (ichunk + 1) * chunk_size] = chol_vecs
            endw = time.time()
            if verbose:
                print(f"# Writing Cholesky chunk {ichunk} to file")
                print(f"# Time to write {endw - startw}")
            ichunk += 1
            chol_vecs[:] = 0.0
        chol_vecs[(nchol + 1) % chunk_size] = (Munu0 - R) / (delta_max) ** 0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            # info = (nchol, delta_max, step_time, endr-startr)
            print(
                "iteration {:5d} : delta_max = {:13.8e} : step time ="
                " {:13.8e} : res time = {:13.8e} ".format(
                    nchol, delta_max, step_time, endr - startr
                )
            )
    with h5py.File(filename, "r+") as fh5:
        fh5["dims"] = np.array([nao * nao, nchol])
    return nchol


def get_ortho_ao(S, LINDEP_CUTOFF=0):
    """Generate canonical orthogonalization transformation matrix.

    Parameters
    ----------
    S : :class:`np.ndarray`
        Overlap matrix.
    LINDEP_CUTOFF : float
        Linear dependency cutoff. Basis functions whose eigenvalues lie below
        this value are removed from the basis set. Should be set in accordance
        with value in pyscf (pyscf.scf.addons.remove_linear_dep_).

    Returns
    -------
    X : :class:`np.array`
        Transformation matrix.
    """
    sdiag, Us = np.linalg.eigh(S)
    X = Us[:, sdiag > LINDEP_CUTOFF] / np.sqrt(sdiag[sdiag > LINDEP_CUTOFF])
    return X


def load_from_pyscf_chkfile(chkfile, base="scf"):
    mol = lib.chkfile.load_mol(chkfile)
    with h5py.File(chkfile, "r") as fh5:
        try:
            hcore = fh5["/scf/hcore"][:]
        except KeyError:
            hcore = scf.hf.get_hcore(mol)
        try:
            X = fh5["/scf/orthoAORot"][:]
        except KeyError:
            s1e = mol.intor("int1e_ovlp_sph")
            X = get_ortho_ao(s1e)
        if base == "mcscf":
            try:
                ci_coeffs = fh5["mcscf/ci_coeffs"][:]
                occa = fh5["mcscf/occs_alpha"][:]
                occb = fh5["mcscf/occs_beta"][:]
            except KeyError:
                ci_coeffs = None
                occa = None
                occb = None
    mo_occ = lib.chkfile.load(chkfile, base + "/mo_occ")
    mo_coeff = lib.chkfile.load(chkfile, base + "/mo_coeff")
    if mo_coeff is None:
        mo_occ = lib.chkfile.load(chkfile, "/scf" + "/mo_occ")
        mo_coeff = lib.chkfile.load(chkfile, "/scf" + "/mo_coeff")
    scf_data = {"mol": mol, "mo_occ": mo_occ, "hcore": hcore, "X": X, "mo_coeff": mo_coeff}
    if base == "mcscf":
        scf_data["ci_coeffs"] = ci_coeffs
        scf_data["occa"] = occa
        scf_data["occb"] = occb
    return scf_data


def freeze_core(h1e, chol, ecore, X, nfrozen, verbose=False):
    # 1. Construct one-body hamiltonian
    nbasis = h1e.shape[-1]
    nchol = chol.shape[0]
    chol = chol.reshape((nchol, nbasis, nbasis))
    ham = dotdict(
        {
            "H1": np.array([h1e, h1e]),
            "chol_vecs": chol.T.copy().reshape((nbasis * nbasis, nchol)),
            "nchol": nchol,
            "ecore": ecore,
            "nbasis": nbasis,
        }
    )
    system = dotdict({"nup": 0, "ndown": 0})
    if len(X.shape) == 2:
        psi_a = np.identity(nbasis)[:, :nfrozen]
        psi_b = np.identity(nbasis)[:, :nfrozen]
    elif len(X.shape) == 3:
        C = X
        psi_a = np.identity(nbasis)[:, :nfrozen]
        Xinv = scipy.linalg.inv(X[0])
        psi_b = np.dot(Xinv, C[1])[:, :nfrozen]

    Gcore_a = gab(psi_a, psi_a)
    Gcore_b = gab(psi_b, psi_b)
    ecore = local_energy_generic_cholesky(system, ham, [Gcore_a, Gcore_b])[0]

    (hc_a, hc_b) = core_contribution_cholesky(chol, [Gcore_a, Gcore_b])
    h1e = np.array([h1e, h1e])
    h1e[0] = h1e[0] + 2 * hc_a
    h1e[1] = h1e[1] + 2 * hc_b
    h1e = h1e[:, nfrozen:, nfrozen:]
    nchol = chol.shape[0]
    nact = h1e.shape[-1]
    chol = chol[:, nfrozen:, nfrozen:].reshape((nchol, nact, nact))
    # 4. Subtract one-body term from writing H2 as sum of squares.
    if verbose:
        print(f"# Number of active orbitals: {nact}")
        print(f"# Freezing {nfrozen} core orbitals.")
        print(f"# Frozen core energy : {ecore.real:15.12e}")
    return h1e, chol, ecore.real


def integrals_from_scf(mf, chol_cut=1e-5, verbose=0, ortho_ao=False):
    mol = mf.mol
    hcore = mf.get_hcore()
    if ortho_ao:
        s1e = mf.mol.intor("int1e_ovlp_sph")
        X = get_ortho_ao(s1e)
    else:
        X = mf.mo_coeff
        if len(X.shape) == 3:
            X = X[0]
    h1e, chol, enuc = generate_integrals(mol, hcore, X, chol_cut=chol_cut, verbose=verbose)
    return h1e, chol, enuc, X


def integrals_from_chkfile(chkfile, chol_cut=1e-5, verbose=False, ortho_ao=False):
    scf_data = load_from_pyscf_chkfile(chkfile)
    mol = scf_data["mol"]
    hcore = scf_data["hcore"]
    if ortho_ao:
        raise ValueError("ortho_ao isn't used")
    else:
        X = scf_data["mo_coeff"]
        if len(X.shape) == 3:
            X = X[0]
    h1e, chol, enuc = generate_integrals(mol, hcore, X, chol_cut=chol_cut, verbose=verbose)
    return h1e, chol, enuc, X
