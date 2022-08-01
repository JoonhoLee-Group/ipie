"""Generate AFQMC data from PYSCF (molecular) simulation."""
import time

import h5py
import numpy
import scipy.linalg
from typing import Union, Tuple

from pyscf import ao2mo, fci, lib, scf, mcscf
from pyscf.tools import fcidump

from ipie.utils.io import write_wavefunction, write_hamiltonian
from ipie.utils.misc import dotdict


def gen_ipie_input_from_pyscf_chk(
        pyscf_chkfile: str,
        hamil_file: str="hamiltonian.h5",
        wfn_file: str="wavefunction.h5",
        verbose: bool=True,
        chol_cut: float=1e-5,
        ortho_ao: bool=False,
        linear_dep_thresh: float=1e-8,
        num_frozen_core: int=0,
) -> None:
    scf_data = load_from_pyscf_chkfile(pyscf_chkfile)
    mol = scf_data["mol"]
    hcore = scf_data["hcore"]
    mo_coeff = scf_data["mo_coeff"]
    uhf = isinstance(mo_coeff, list)
    if uhf:
        if ortho_ao:
            X = get_ortho_ao(mol.intor('s1e_ovlp_sph'), linear_dep_thresh)
        else:
            X = mo_coeff[0]
    else:
        X = mo_coeff

    mcscf_dict = scf_data.get('mcscf')
    if mcscf_dict is not None:
        if num_melting == 0:
            hcore, chol, e0 = generate_integrals(mol, hcore, X, chol_cut=1e-5, verbose=False, cas=None)
        else:
            pass
        write_hamiltonian(hcore, chol, enuc, filename=hamil_file)
        ci_coeffs = mcscf_dict['ci_coeffs']
        occa0 = mcscf_dict['occa']
        occb0 = mcscf_dict['occb']
        occa, occb = insert_melting_core(occa0, occb0, num_melting)
        write_wavefunction((ci_coeffs, occa, occb), filename=wfn_file)
    else:
        hcore, chol, e0 = generate_integrals(mol, hcore, X, chol_cut=1e-5, verbose=False, cas=None)
        write_hamiltonian(hcore, chol, e0, filename=hamil_file)
        write_wavefunction_from_mo_coeff(mo_coeff, X, wfn_file, mol.nelec)


def write_wavefunction_from_mo_coeff(
        mo_coeff: Union[list, numpy.ndarray],
        X: numpy.ndarray,
        filename: str,
        nelec: tuple,
        ortho_ao: bool=False,
        num_melting_core: int=0
) -> None:
    """Generate QMCPACK trial wavefunction.
    """
    uhf = isinstance(mo_coeff, list)
    norb = X.shape[1]
    nalpha, nbeta = nelec
    if ortho_ao:
        Xinv = scipy.linalg.inv(X)
        if uhf:
            # We are assuming C matrix is energy ordered.
            wfna = numpy.dot(Xinv, C[0])[:, :nalpha]
            wfnb = numpy.dot(Xinv, C[1])[:, :nbeta]
            write_wavefunction([wfna, wfnb], filename=filename)
        else:
            wfna = numpy.dot(Xinv, C)[:, :nalpha]
            write_wavefunction(wfna, filename=filename)
    else:
        if uhf:
            # HP: Assuming we are working in the alpha orbital basis, and write the beta orbitals as LCAO of alpha orbitals
            I = numpy.identity(norb, dtype=numpy.float64)
            wfna = I[:, :nalpha]
            Xinv = scipy.linalg.inv(X)
            wfnb = numpy.dot(Xinv, mo_coeff[1])[:, :nbeta]
            write_wavefunction([wfna, wfnb], filename=filename)
        else:
            # Assuming we are working in MO basis, only works for RHF, ROHF trials.
            I = numpy.identity(norb, dtype=numpy.float64)
            wfna = I[:, :nalpha]
            write_wavefunction(wfna, filename=filename)


def generate_integrals(mol, hcore, X, chol_cut=1e-5, verbose=False, cas=None):
    # Unpack SCF data.
    # Step 1. Rotate core Hamiltonian to orthogonal basis.
    if verbose:
        print(" # Transforming hcore and eri to ortho AO basis.")
    if len(X.shape) == 2:
        h1e = numpy.dot(X.T, numpy.dot(hcore, X))
    elif len(X.shape) == 3:
        h1e = numpy.dot(X[0].T, numpy.dot(hcore, X[0]))
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
        print(" # Time to orthogonalise: %f" % (time.time() - start))
    enuc = mol.energy_nuc()
    # Step 3. (Optionally) freeze core / virtuals.
    nelec = mol.nelec

    return h1e, chol_vecs.reshape((-1, nbasis, nbasis)), enuc


def ao2mo_chol(eri, C, verbose=False):
    nb = C.shape[-1]
    for i, cv in enumerate(eri):
        if verbose and i % 100 == 0:
            print(
                " # ao2mo cholesky % complete = {} %".format(100 * float(i) / len(eri))
            )
        half = numpy.dot(cv.reshape(nb, nb), C)
        eri[i] = numpy.dot(C.conj().T, half).ravel()


def cholesky(
    mol,
    filename="hamil.h5",
    max_error=1e-6,
    verbose=False,
    cmax=20,
    CHUNK_SIZE=2.0,
    MAX_SIZE=20.0,
):
    nao = mol.nao_nr()
    if nao * nao * cmax * nao * 8.0 / 1024.0**3 > MAX_SIZE:
        if verbose:
            print(
                "# Approximate memory for cholesky > MAX_SIZE ({} GB).".format(MAX_SIZE)
            )
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
            return chunked_cholesky(
                mol, max_error=max_error, verbose=verbose, cmax=cmax
            )


def chunked_cholesky(mol, max_error=1e-6, verbose=False, cmax=10):
    """Modified cholesky decomposition from pyscf eris.

    See, e.g. [Motta17]_

    Only works for molecular systems.

    Parameters
    ----------
    mol : :class:`pyscf.mol`
        pyscf mol object.
    orthoAO: :class:`numpy.ndarray`
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
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = numpy.zeros(nao * nao)
    nchol_max = cmax * nao
    # This shape is more convenient for ipie.
    chol_vecs = numpy.zeros((nchol_max, nao * nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0, mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2 * l + 1) * nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0, mol.nbas):
        shls = (i, i + 1, 0, mol.nbas, i, i + 1, 0, mol.nbas)
        buf = mol.intor("int2e_sph", shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = numpy.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs." % nchol_max)
        print("# max number of cholesky vectors = %d" % nchol_max)
        print("# iteration %5d: delta_max = %f" % (0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = numpy.searchsorted(dims, j)
    sl = numpy.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = numpy.zeros(nao * nao)
    # ERI[:,jl]
    eri_col = mol.intor(
        "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
    )
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = (
        numpy.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5
    )

    nchol = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol] * chol_vecs[nchol]
        # D_ii = M_ii - M'_ii
        delta = diag - Mapprox
        nu = numpy.argmax(numpy.abs(delta))
        delta_max = numpy.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = numpy.searchsorted(dims, j)
        sl = numpy.searchsorted(dims, l)
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
        R = numpy.dot(chol_vecs[: nchol + 1, nu], chol_vecs[: nchol + 1, :])
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
    orthoAO: :class:`numpy.ndarray`
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
    chol_vecs : :class:`numpy.ndarray`
        Matrix of cholesky vectors in AO basis.
    """
    nao = mol.nao_nr()
    diag = numpy.zeros(nao * nao)
    nchol_max = cmax * nao
    mem = 8.0 * nchol_max * nao * nao / 1024.0**3
    chunk_size = min(int(CHUNK_SIZE * 1024.0**3 / (8 * nao * nao)), nchol_max)
    if verbose:
        print("# Number of AOs: {}".format(nao))
        print("# Writing AO Cholesky to {:s}.".format(filename))
        print("# Max number of Cholesky vectors: {}".format(nchol_max))
        print("# Max memory required for Cholesky tensor: {} GB".format(mem))
        print(
            "# Splitting calculation into chunks of size: {} / GB".format(
                chunk_size, 8 * chunk_size * nao * nao / (1024.0**3)
            )
        )
        print("# Generating diagonal.")
    chol_vecs = numpy.zeros((chunk_size, nao * nao))
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
        di, dk, dj, dl = buf.shape
        diag[ndiag : ndiag + di * nao] = buf.reshape(di * nao, di * nao).diagonal()
        ndiag += di * nao
    nu = numpy.argmax(diag)
    delta_max = diag[nu]
    with h5py.File(filename, "w") as fh5:
        fh5.create_dataset("Lao", shape=(nchol_max, nao * nao), dtype=numpy.float64)
    end = time.time()
    if verbose:
        print("# Time to generate diagonal {} s.".format(end - start))
        print("# Generating Cholesky decomposition of ERIs.")
        print("# iteration {:5d}: delta_max = {:13.8e}".format(0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = numpy.searchsorted(dims, j)
    sl = numpy.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = numpy.zeros(nao * nao)
    eri_col = mol.intor(
        "int2e_sph", shls_slice=(0, mol.nbas, 0, mol.nbas, sj, sj + 1, sl, sl + 1)
    )
    cj, cl = max(j - dims[sj], 0), max(l - dims[sl], 0)
    chol_vecs[0] = (
        numpy.copy(eri_col[:, :, cj, cl].reshape(nao * nao)) / delta_max**0.5
    )

    def compute_residual(chol, ichol, nchol, nu):
        # Updated residual = \sum_x L_i^x L_nu^x
        # R = numpy.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        R = 0.0
        with h5py.File(filename, "r") as fh5:
            for ic in range(0, ichol):
                # Compute dot product from file.
                # print(ichol*chunk_size, (ichol+1)*chunk_size)
                # import sys
                # sys.exit()
                # print(ic*chunk_size, (ic*chunk_size)
                L = fh5["Lao"][ic * chunk_size : (ic + 1) * chunk_size, :]
                R += numpy.dot(L[:, nu], L[:, :])
        R += numpy.dot(chol[: nchol + 1, nu], chol[: nchol + 1, :])
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
        nu = numpy.argmax(numpy.abs(delta))
        delta_max = numpy.abs(delta[nu])
        # Compute ERI chunk.
        # shls_slice computes shells of integrals as determined by the angular
        # momentum of the basis function and the number of contraction
        # coefficients. Need to search for AO index within this shell indexing
        # scheme.
        # AO index.
        j = nu // nao
        l = nu % nao
        # Associated shell index.
        sj = numpy.searchsorted(dims, j)
        sl = numpy.searchsorted(dims, l)
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
                print("# Writing Cholesky chunk {} to file".format(ichunk))
                print("# Time to write {}".format(endw - startw))
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
        fh5["dims"] = numpy.array([nao * nao, nchol])
    return nchol


def get_ortho_ao(S, LINDEP_CUTOFF=0):
    """Generate canonical orthogonalization transformation matrix.

    Parameters
    ----------
    S : :class:`numpy.ndarray`
        Overlap matrix.
    LINDEP_CUTOFF : float
        Linear dependency cutoff. Basis functions whose eigenvalues lie below
        this value are removed from the basis set. Should be set in accordance
        with value in pyscf (pyscf.scf.addons.remove_linear_dep_).

    Returns
    -------
    X : :class:`numpy.array`
        Transformation matrix.
    """
    sdiag, Us = numpy.linalg.eigh(S)
    X = Us[:, sdiag > LINDEP_CUTOFF] / numpy.sqrt(sdiag[sdiag > LINDEP_CUTOFF])
    return X


def load_from_pyscf_chkfile(chkfile, base="scf"):
    mol = lib.chkfile.load_mol(chkfile)
    with h5py.File(chkfile, "r") as fh5:
        try:
            hcore = fh5["/scf/hcore"][:]
        except KeyError:
            hcore = mol.intor_symmetric("int1e_nuc")
            hcore += mol.intor_symmetric("int1e_kin")
            if len(mol._ecpbas) > 0:
                hcore += mol.intor_symmetric("ECPScalar")
        try:
            X = fh5["/scf/orthoAORot"][:]
        except KeyError:
            s1e = mol.intor("int1e_ovlp_sph")
            X = get_ortho_ao(s1e)
    mo_occ = numpy.array(lib.chkfile.load(chkfile, base + "/mo_occ"))
    mo_coeff = numpy.array(lib.chkfile.load(chkfile, base + "/mo_coeff"))
    scf_data = {
        "mol": mol,
        "mo_occ": mo_occ,
        "hcore": hcore,
        "X": X,
        "mo_coeff": mo_coeff,
    }
    return scf_data
