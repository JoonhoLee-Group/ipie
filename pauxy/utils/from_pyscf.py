"""Generate AFQMC data from PYSCF (molecular) simulation."""
import h5py
import numpy
import time
import scipy.linalg

from pyscf import lib
from pyscf import ao2mo, scf, fci
from pyscf.tools import fcidump

from pauxy.utils.misc import dotdict
from pauxy.utils.io import (
        write_qmcpack_sparse,
        write_qmcpack_dense,
        write_qmcpack_wfn
        )
from pauxy.estimators.greens_function import gab
from pauxy.estimators.generic import (
        local_energy_generic_cholesky, core_contribution_cholesky
    )

def dump_pauxy(chkfile=None, mol=None, mf=None, hamil_file='afqmc.h5',
               verbose=True, wfn_file='afqmc.h5',
               chol_cut=1e-5, sparse_zero=1e-16, cas=None,
               ortho_ao=True, ao=False, sparse=False):
    scf_data = load_from_pyscf_chkfile(chkfile)
    mol = scf_data['mol']
    hcore = scf_data['hcore']
    if ortho_ao:
        oao = scf_data['X']
    else:
        if (ao):
            print(" # Writing everything in the AO basis")
            oao = scf_data['X']
            nbsf = oao.shape[-1]
            if (len(oao.shape) == 3):
                oao[1] = numpy.eye(nbsf)
            else:
                oao = numpy.eye(nbsf)
            scf_data['X'] = oao.copy()
        else:
            oao = scf_data['mo_coeff']

    hcore, chol, nelec, enuc = generate_integrals(mol, hcore, oao,
                                                  chol_cut=chol_cut,
                                                  verbose=verbose,
                                                  cas=cas)
    nbasis = hcore.shape[-1]
    msq = nbasis * nbasis
    # Why did I transpose everything?
    # QMCPACK expects [M^2, N_chol]
    # Internally store [N_chol, M^2]
    chol = chol.T.copy()
    if sparse:
        print(" # Writing integrals in sparse format.")
        write_qmcpack_sparse(hcore, chol, nelec, nbasis, enuc,
                             filename=hamil_file, real_chol=True,
                             verbose=verbose, ortho=oao)
    else:
        print(" # Writing integrals in dense format.")
        write_qmcpack_dense(hcore, chol, nelec,
                            nbasis, enuc,
                            filename=hamil_file,
                            ortho=oao, real_chol=True)
    write_wfn_mol(scf_data, ortho_ao, wfn_file, mode='a')

def write_wfn_mol(scf_data, ortho_ao, filename, wfn=None,
                  init=None, verbose=False, mode='w'):
    """Generate QMCPACK trial wavefunction.

    Parameters
    ----------
    scf_data : dict
        Dictionary containing scf data extracted from pyscf checkpoint file.
    ortho_ao : bool
        Whether we are working in orthogonalised AO basis or not.
    filename : string
        HDF5 file path to store wavefunction to.
    wfn : tuple
        User defined wavefunction. Not fully supported. Default None.

    Returns
    -------
    wfn : :class:`numpy.ndarray`
        Wavefunction as numpy array. Format depends on wavefunction.
    """
    ghf = False
    mol = scf_data['mol']
    nelec = mol.nelec
    nalpha, nbeta = nelec
    C = scf_data['mo_coeff']
    X = scf_data['X']
    uhf = scf_data['isUHF']
    # For RHF only nalpha entries will be filled.
    if uhf:
        norb = C[0].shape[0]
    else:
        norb = C.shape[0]
    if wfn is None:
        wfn = numpy.zeros((1,norb,nalpha+nbeta), dtype=numpy.complex128)
        wfn_type = 'NOMSD'
        coeffs = numpy.array([1.0+0j])
        if ortho_ao:
            Xinv = scipy.linalg.inv(X)
            if uhf:
                # We are assuming C matrix is energy ordered.
                wfn[0,:,:nalpha] = numpy.dot(Xinv, C[0])[:,:nalpha]
                wfn[0,:,nalpha:] = numpy.dot(Xinv, C[1])[:,:nbeta]
            else:
                wfn[0,:,:nalpha] = numpy.dot(Xinv, C)[:,:nalpha]
                wfn[0,:,nalpha:] = numpy.dot(Xinv, C)[:,:nalpha]
        else:
            # Assuming we are working in MO basis, only works for RHF, ROHF trials.
            I = numpy.identity(C.shape[-1], dtype=numpy.float64)
            wfn[0,:,:nalpha] = I[:,:nalpha]
            wfn[0,:,nalpha:] = I[:,:nalpha]
            if uhf:
                print(" # Warning: UHF trial wavefunction can only be used of "
                      "working in ortho AO basis.")
    write_qmcpack_wfn(filename, (numpy.array([1.0+0j]),wfn), 'uhf',
                      nelec, norb, mode=mode)
    return nelec

def integrals_from_scf(mf, chol_cut=1e-5, verbose=0, cas=None, ortho_ao=True):
    mol = mf.mol
    ecore = mf.energy_nuc()
    hcore = mf.get_hcore()
    if ortho_ao:
        s1e = mf.mol.intor('int1e_ovlp_sph')
        X = get_ortho_ao(s1e)
    else:
        X = mf.mo_coeff
    h1e, chol, nelec, enuc = generate_integrals(mol, hcore, X,
                                                chol_cut=chol_cut,
                                                verbose=verbose,
                                                cas=cas)
    return h1e, chol, nelec, enuc

def integrals_from_chkfile(chkfile, chol_cut=1e-5, verbose=False,
                           cas=None, ortho_ao=True):
    scf_data = load_from_pyscf_chkfile(chkfile)
    mol = scf_data['mol']
    hcore = scf_data['hcore']
    if ortho_ao:
        oao = scf_data['X']
    else:
        oao = scf_data['mo_coeff']
    h1e, chol, nelec, enuc = generate_integrals(mol, hcore, oao,
                                                chol_cut=chol_cut,
                                                verbose=verbose,
                                                cas=cas)
    return h1e, chol, nelec, enuc

def generate_integrals(mol, hcore, X, chol_cut=1e-5, verbose=False, cas=None):
    # Unpack SCF data.
    # Step 1. Rotate core Hamiltonian to orthogonal basis.
    if verbose:
        print(" # Transforming hcore and eri to ortho AO basis.")
    if (len(X.shape) == 2):
        h1e = numpy.dot(X.T, numpy.dot(hcore, X))
    elif (len(X.shape) == 3):
        h1e = numpy.dot(X[0].T, numpy.dot(hcore, X[0]))
        
    nbasis = h1e.shape[-1]
    # Step 2. Genrate Cholesky decomposed ERIs in non-orthogonal AO basis.
    if verbose:
        print (" # Performing modified Cholesky decomposition on ERI tensor.")
    chol_vecs = chunked_cholesky(mol, max_error=chol_cut, verbose=verbose)
    if verbose:
        print (" # Orthogonalising Cholesky vectors.")
    start = time.time()
    # Step 2.a Orthogonalise Cholesky vectors.
    if (len(X.shape) == 2):
        ao2mo_chol(chol_vecs, X)
    elif (len(X.shape) == 3):
        ao2mo_chol(chol_vecs, X[0])
    if verbose:
        print (" # Time to orthogonalise: %f"%(time.time() - start))
    enuc = mol.energy_nuc()
    # Step 3. (Optionally) freeze core / virtuals.
    nelec = mol.nelec
    if cas is not None:
        nfzc = (sum(mol.nelec)-cas[0])//2
        ncas = cas[1]
        nfzv = nbasis - ncas - nfzc
        h1e, chol_vecs, enuc = freeze_core(h1e, chol_vecs, enuc, nfzc, ncas,
                                           verbose)
        h1e = h1e[0]
        nelec = (mol.nelec[0]-nfzc, mol.nelec[1]-nfzc)
        mol.nelec = nelec
        orbs = numpy.identity(h1e.shape[-1])
        orbs = orbs[nfzc:nbasis-nfzv,nfzc:nbasis-nfzv]
    return h1e, chol_vecs, nelec, enuc

def freeze_core(h1e, chol, ecore, nc, ncas, verbose=True):
    # 1. Construct one-body hamiltonian
    print(ecore, type(h1e), type(chol))
    nbasis = h1e.shape[-1]
    chol = chol.reshape((-1,nbasis,nbasis))
    system = dotdict({'H1': numpy.array([h1e,h1e]),
                      'chol_vecs': chol,
                      'ecore': ecore,
                      'nbasis': nbasis})
    psi = numpy.identity(nbasis)[:,:nc]
    Gcore = gab(psi,psi)
    ecore = local_energy_generic_cholesky(system, [Gcore,Gcore])[0]
    (hc_a, hc_b) = core_contribution_cholesky(system.chol_vecs, [Gcore,Gcore])
    h1e = numpy.array([h1e,h1e])
    h1e[0] = h1e[0] + 2*hc_a
    h1e[1] = h1e[1] + 2*hc_b
    h1e = h1e[:,nc:nc+ncas,nc:nc+ncas]
    nchol = chol.shape[0]
    chol = chol[:,nc:nc+ncas,nc:nc+ncas].reshape((nchol,-1))
    # 4. Subtract one-body term from writing H2 as sum of squares.
    if verbose:
        print(" # Number of active orbitals: %d"%ncas)
        print(" # Freezing %d core electrons and %d virtuals."
              %(2*nc, nbasis-nc-ncas))
        print(" # Frozen core energy : %13.8e"%ecore.real)
    return h1e, chol, ecore

def ao2mo_chol(eri, C, verbose=False):
    nb = C.shape[-1]
    for i, cv in enumerate(eri):
        if verbose and i % 100 == 0:
            print(" # ao2mo cholesky % complete = {} %".format(100*float(i)/len(eri)))
        half = numpy.dot(cv.reshape(nb,nb), C)
        eri[i] = numpy.dot(C.conj().T, half).ravel()

def load_from_pyscf_chkfile(chkfile, base='scf'):
    mol = lib.chkfile.load_mol(chkfile)
    with h5py.File(chkfile, 'r') as fh5:
        try:
            hcore = fh5['/scf/hcore'][:]
        except KeyError:
            hcore = mol.intor_symmetric('int1e_nuc')
            hcore += mol.intor_symmetric('int1e_kin')
            if len(mol._ecpbas) > 0:
                hcore += mol.intor_symmetric('ECPScalar')
        try:
            X = fh5['/scf/orthoAORot'][:]
        except KeyError:
            s1e = mol.intor('int1e_ovlp_sph')
            X = get_ortho_ao(s1e)
    mo_occ = numpy.array(lib.chkfile.load(chkfile, base+'/mo_occ'))
    mo_coeff = numpy.array(lib.chkfile.load(chkfile, base+'/mo_coeff'))
    uhf = len(mo_coeff.shape) == 3
    scf_data = {'mol': mol, 'mo_occ': mo_occ, 'hcore': hcore,
                'X': X, 'mo_coeff': mo_coeff,
                'isUHF': uhf}
    return scf_data

def from_pyscf_scf(mf, verbose=True):
    hcore = mf.get_hcore()
    fock = hcore + mf.get_veff()
    s1e = mf.mol.intor('int1e_ovlp_sph')
    orthoAO = get_ortho_ao(s1e)
    enuc = mf.energy_nuc()
    if verbose:
        print ("# Generating PAUXY input PYSCF mol and scf objects.")
        print ("# (nalpha, nbeta): (%d, %d)"%mf.mol.nelec)
        print ("# nbasis: %d"%hcore.shape[-1])
    return (hcore, fock, orthoAO, enuc)

def write_fcidump(system, name='FCIDUMP'):
    fcidump.from_integrals(name, system.H1[0], system.h2e,
                           system.H1[0].shape[0], system.ne, nuc=system.ecore)

def cholesky(mol, filename='hamil.h5', max_error=1e-6, verbose=False, cmax=20,
             CHUNK_SIZE=2.0, MAX_SIZE=20.0):
    nao = mol.nao_nr()
    if nao*nao*cmax*nao*8.0 / 1024.0**3 > MAX_SIZE:
        if verbose:
            print("# Approximate memory for cholesky > MAX_SIZE ({} GB)."
                  .format(MAX_SIZE))
            print("# Using out of core algorithm.")
            return chunked_cholesky_outcore(mol, filename=filename,
                                            max_error=max_error,
                                            verbose=verbose,
                                            cmax=cmax,
                                            CHUNK_SIZE=CHUNK_SIZE)
        else:
            return chunked_cholesky(mol, max_error=max_error, verbose=verbose,
                                    cmax=cmax)

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
    diag = numpy.zeros(nao*nao)
    nchol_max = cmax * nao
    # This shape is more convenient for pauxy.
    chol_vecs = numpy.zeros((nchol_max, nao*nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    for i in range(0,mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2*l+1)*nc
        dims.append(nao_per_i)
    # print (dims)
    for i in range(0,mol.nbas):
        shls = (i,i+1,0,mol.nbas,i,i+1,0,mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag+di*nao] = buf.reshape(di*nao,di*nao).diagonal()
        ndiag += di * nao
    nu = numpy.argmax(diag)
    delta_max = diag[nu]
    if verbose:
        print("# Generating Cholesky decomposition of ERIs."%nchol_max)
        print("# max number of cholesky vectors = %d"%nchol_max)
        print("# iteration %5d: delta_max = %f"%(0, delta_max))
    j = nu // nao
    l = nu % nao
    sj = numpy.searchsorted(dims, j)
    sl = numpy.searchsorted(dims, l)
    if dims[sj] != j and j != 0:
        sj -= 1
    if dims[sl] != l and l != 0:
        sl -= 1
    Mapprox = numpy.zeros(nao*nao)
    # ERI[:,jl]
    eri_col = mol.intor('int2e_sph',
                         shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
    cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
    chol_vecs[0] = numpy.copy(eri_col[:,:,cj,cl].reshape(nao*nao)) / delta_max**0.5

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
        eri_col = mol.intor('int2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
        Munu0 = eri_col[:,:,cj,cl].reshape(nao*nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        R = numpy.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        chol_vecs[nchol+1] = (Munu0 - R) / (delta_max)**0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            info = (nchol, delta_max, step_time)
            print ("# iteration %5d: delta_max = %13.8e: time = %13.8e"%info)

    return chol_vecs[:nchol]

def chunked_cholesky_outcore(mol, filename='hamil.h5', max_error=1e-6,
                             verbose=False, cmax=20, CHUNK_SIZE=2.0):
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
    diag = numpy.zeros(nao*nao)
    nchol_max = cmax * nao
    mem = 8.0*nchol_max*nao*nao / 1024.0**3
    chunk_size = min(int(CHUNK_SIZE*1024.0**3/(8*nao*nao)),nchol_max)
    if verbose:
        print("# Number of AOs: {}".format(nao))
        print("# Writing AO Cholesky to {:s}.".format(filename))
        print("# Max number of Cholesky vectors: {}".format(nchol_max))
        print("# Max memory required for Cholesky tensor: {} GB".format(mem))
        print("# Splitting calculation into chunks of size: {} / GB"
              .format(chunk_size, 8*chunk_size*nao*nao/(1024.0**3)))
        print("# Generating diagonal.")
    chol_vecs = numpy.zeros((chunk_size,nao*nao))
    ndiag = 0
    dims = [0]
    nao_per_i = 0
    start = time.time()
    for i in range(0,mol.nbas):
        l = mol.bas_angular(i)
        nc = mol.bas_nctr(i)
        nao_per_i += (2*l+1)*nc
        dims.append(nao_per_i)
    for i in range(0,mol.nbas):
        shls = (i,i+1,0,mol.nbas,i,i+1,0,mol.nbas)
        buf = mol.intor('int2e_sph', shls_slice=shls)
        di, dk, dj, dl = buf.shape
        diag[ndiag:ndiag+di*nao] = buf.reshape(di*nao,di*nao).diagonal()
        ndiag += di * nao
    nu = numpy.argmax(diag)
    delta_max = diag[nu]
    with h5py.File(filename, 'w') as fh5:
        fh5.create_dataset('Lao',
                           shape=(nchol_max, nao*nao),
                           dtype=numpy.float64)
    end = time.time()
    if verbose:
        print("# Time to generate diagonal {} s.".format(end-start))
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
    Mapprox = numpy.zeros(nao*nao)
    eri_col = mol.intor('int2e_sph',
                        shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
    cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
    chol_vecs[0] = numpy.copy(eri_col[:,:,cj,cl].reshape(nao*nao)) / delta_max**0.5

    def compute_residual(chol, ichol, nchol, nu):
        # Updated residual = \sum_x L_i^x L_nu^x
        # R = numpy.dot(chol_vecs[:nchol+1,nu], chol_vecs[:nchol+1,:])
        R = 0.0
        with h5py.File(filename, 'r') as fh5:
            for ic in range(0, ichol):
                # Compute dot product from file.
                # print(ichol*chunk_size, (ichol+1)*chunk_size)
                # import sys
                # sys.exit()
                # print(ic*chunk_size, (ic*chunk_size)
                L = fh5['Lao'][ic*chunk_size:(ic+1)*chunk_size,:]
                R += numpy.dot(L[:,nu], L[:,:])
        R += numpy.dot(chol[:nchol+1,nu], chol[:nchol+1,:])
        return R

    nchol = 0
    ichunk = 0
    while abs(delta_max) > max_error:
        # Update cholesky vector
        start = time.time()
        # M'_ii = L_i^x L_i^x
        Mapprox += chol_vecs[nchol%chunk_size] * chol_vecs[nchol%chunk_size]
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
        eri_col = mol.intor('int2e_sph',
                            shls_slice=(0,mol.nbas,0,mol.nbas,sj,sj+1,sl,sl+1))
        # Select correct ERI chunk from shell.
        cj, cl = max(j-dims[sj],0), max(l-dims[sl],0)
        Munu0 = eri_col[:,:,cj,cl].reshape(nao*nao)
        # Updated residual = \sum_x L_i^x L_nu^x
        startr = time.time()
        R = compute_residual(chol_vecs, ichunk, nchol%chunk_size, nu)
        endr = time.time()
        if nchol > 0 and (nchol + 1) % chunk_size == 0:
            startw = time.time()
            # delta = L[ichunk*chunk_size:(ichunk+1)*chunk_size]-chol_vecs
            with h5py.File(filename, 'r+') as fh5:
                fh5['Lao'][ichunk*chunk_size:(ichunk+1)*chunk_size] = chol_vecs
            endw = time.time()
            if verbose:
                print("# Writing Cholesky chunk {} to file".format(ichunk))
                print("# Time to write {}".format(endw-startw))
            ichunk += 1
            chol_vecs[:] = 0.0
        chol_vecs[(nchol+1)%chunk_size] = (Munu0 - R) / (delta_max)**0.5
        nchol += 1
        if verbose:
            step_time = time.time() - start
            # info = (nchol, delta_max, step_time, endr-startr)
            print("iteration {:5d} : delta_max = {:13.8e} : step time ="
                  " {:13.8e} : res time = {:13.8e} "
                  .format(nchol, delta_max, step_time, endr-startr))
    with h5py.File(filename, 'r+') as fh5:
        fh5['dims'] = numpy.array([nao*nao, nchol])
    return nchol


def multi_det_wavefunction(mc, weight_cutoff=0.95, verbose=False,
                           max_ndets=1e5, norb=None,
                           filename="multi_det.dat"):
    """Generate multi determinant particle-hole trial wavefunction.

    Format adopted to be compatable with QMCPACK PHMSD type wavefunction.

    Parameters
    ----------
    mc : pyscf CI solver type object
        Input object containing multi determinant coefficients.
    weight_cutoff : float, optional
        Print determinants until accumulated weight equals weight_cutoff.
        Default 0.95.
    verbose : bool
        Print information about process. Default False.
    max_ndets : int
        Max number of determinants to print out. Default 1e5.
    norb : int or None, optional
        Total number of orbitals in simulation. Used if we want to run CI within
        active space but QMC in full space. Deault None.
    filename : string
        Output filename. Default "multi_det.dat"
    """
    occlists = fci.cistring._gen_occslst(range(mc.ncas), mc.nelecas[0])

    ci_coeffs = mc.ci.ravel()
    # Sort coefficients in terms of increasing absolute weight.
    ix_sort = numpy.argsort(numpy.abs(ci_coeffs))[::-1]
    cweight = numpy.cumsum(ci_coeffs[ix_sort]**2)
    max_det = numpy.searchsorted(cweight, weight_cutoff)
    ci_coeffs = ci_coeffs[ix_sort]
    if verbose:
        print("Number of dets in CAS space: %d"%len(occlists)**2)
        print("Number of dets in CI expansion: %d"%max_det)

    output = open(filename, 'w')
    namelist = "&FCI\n UHF = 0\n NCI = %d\n TYPE = occ\n&END" % max_det
    output.write(namelist+'\n')
    output.write("Configurations:"+'\n')
    if norb is None:
        norb = mc.ncas

    for idet in range(max_det):
        if mc.ncore > 0:
            ocore_up = ' '.join('{:d}'.format(x+1) for x in range(mc.ncore))
            ocore_dn = ' '.join('{:d}'.format(x+1+norb) for x in range(mc.ncore))
        else:
            ocore_up = ' '
            ocore_dn = ' '
        coeff = '%.13f'%ci_coeffs[idet]
        ix_alpha = ix_sort[idet] // len(occlists)
        ix_beta = ix_sort[idet] % len(occlists)
        ia = occlists[ix_alpha]
        ib = occlists[ix_beta]
        oup = ' '.join('{:d}'.format(x+1+mc.ncore) for x in ia)
        odown = ' '.join('{:d}'.format(x+norb+1+mc.ncore) for x in ib)
        output.write(coeff+' '+ocore_up+' '+oup+' '+ocore_dn+' '+odown+'\n')

def get_pyscf_wfn(system, mf):
    """Return trial wavefunction from pyscf mf object.
    """
    C = mf.mo_coeff
    na = system.nup
    nb = system.ndown
    X = system.oao
    Xinv = scipy.linalg.inv(X)
    # TODO : Update for mcscf object.
    if len(C.shape) == 3:
        # UHF trial.
        pa = numpy.dot(Xinv, C[0][:,:na])
        pb = numpy.dot(Xinv, C[1][:,:nb])
    else:
        pa = numpy.dot(Xinv, C[:,:na])
        pb = pa.copy()
    wfn = numpy.zeros((system.nbasis, na+nb), dtype=numpy.complex128)
    wfn[:,:na] = pa
    wfn[:,na:] = pb
    return (numpy.array([1.0+0j]), wfn)

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
    X = Us[:,sdiag>LINDEP_CUTOFF] / numpy.sqrt(sdiag[sdiag>LINDEP_CUTOFF])
    return X
