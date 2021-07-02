import ast
import h5py
import numpy
import sys
import scipy.linalg
import time
from scipy.sparse import csr_matrix
from pauxy.utils.linalg import modified_cholesky
from pauxy.utils.io import (
        from_qmcpack_sparse,
        from_qmcpack_dense,
        write_qmcpack_sparse,
        write_qmcpack_dense,
        )
from pauxy.estimators.generic import (
        local_energy_generic, core_contribution,
        local_energy_generic_cholesky, core_contribution_cholesky
)
from pauxy.utils.mpi import get_shared_array, have_shared_mem


class Generic(object):
    """Generic system defined by ab-initio Hamiltonian.

    Can be created by either passing the one and two electron integrals directly
    or initialised from integrals stored in QMCPACK hdf5 format. If initialising
    from file the `inputs' optional dictionary should be populated.

    Parameters
    ----------
    nelec : tuple
        Number of alpha and beta electrons.
    h1e : :class:`numpy.ndarray'
        One-electron integrals. Optional. Default: None.
    chol : :class:`numpy.ndarray'
        Factorized 2-electron integrals (L_{ik,n}) of shape (nbasis^2, nchol).
        Optional. Default: None.
    ecore : float
        Core energy.
    inputs : dict
        Input options defined below.
    nup : int
        Number of up electrons.
    ndown : int
        Number of down electrons.
    integrals : string
        Path to file containing one- and two-electron integrals in QMCPACK
        format.
    verbose : bool
        Print extra information.

    Attributes
    ----------
    H1 : :class:`numpy.ndarray`
        One-body part of the Hamiltonian. Spin-dependent by default.
    ecore : float
        Core contribution to the total energy.
    h1e_mod : :class:`numpy.ndarray`
        Modified one-body Hamiltonian.
    chol_vecs : :class:`numpy.ndarray`
        Cholesky vectors. [M^2, nchol]
    nchol : int
        Number of cholesky vectors.
    nfields : int
        Number of auxiliary fields required.
    sparse_cutoff : float
        Screen out integrals below this threshold. Optional. Default 0.
    cplx_chol : bool
        Force setting of interpretation of cholesky decomposition. Optional.
        Default False, i.e. real/complex factorization determined from cholesky
        integrals.
    """

    def __init__(self, nelec=None, h1e=None, chol=None, ecore=None, h1e_mod=None,
                 mu=None, verbose=False, write_ints=False,
                 stochastic_ri=False,
                 exact_eri=False,
                 nsamples=0,
                 pno=False,
                 thresh_pno=None,
                 control_variate=False):
        if verbose:
            print("# Parsing input options.")
        self.name = "Generic"
        self.verbose = verbose
        self.nup, self.ndown = nelec
        self.nelec = nelec
        self.ne = self.nup + self.ndown
        self.stochastic_ri = stochastic_ri
        self.pno = pno

        self.thresh_pno = thresh_pno

        if (self.pno):
            self.ij_list_aa = []
            self.ij_list_bb = []
            self.ij_list_ab = []
            
            for i in range(self.nup):
                for j in range(i, self.nup):
                    self.ij_list_aa += [(i,j)]
            
            for i in range(self.ndown):
                for j in range(i, self.ndown):
                    self.ij_list_bb += [(i,j)]
            
            for i in range(self.nup):
                for j in range(self.ndown):
                    self.ij_list_ab += [(i,j)]

        self.exact_eri = exact_eri
        self.control_variate = control_variate
        if self.exact_eri:
            if self.verbose:
                print("# exact_eri is true for local energy")
        if self.pno:
            if self.verbose:
                print("# pno is true for local energy with a threshold of {}".format(self.thresh_pno))
        if self.stochastic_ri:
            self.nsamples = nsamples
            if self.verbose:
                print("# stochastic_ri is true for local energy with {} samples".format(self.nsamples))
                print("# control_variate = {}".format(self.control_variate))
        self.mu = mu
        self.ecore = ecore
        self.chol_vecs = chol
        if isinstance(self.chol_vecs.dtype, numpy.complex128):
            if verbose:
                print("# Found complex integrals.")
                print("# Using Hermitian Cholesky decomposition.")
        else:
            if verbose:
                print("# Using real Cholesky decomposition.")
            self.cplx_chol = False
        self.H1 = h1e
        self.nbasis = h1e.shape[-1]
        self._alt_convention = False
        mem = self.chol_vecs.nbytes / (1024.0**3)
        self.sparse = False
        if verbose:
            print("# Number of orbitals: %d"%self.nbasis)
            print("# Number of electrons: (%d, %d)"%(self.nup, self.ndown))
            print("# Approximate memory required by Cholesky vectors %f GB"%mem)
        self.nchol = self.chol_vecs.shape[-1]
        if h1e_mod is not None:
            self.h1e_mod = h1e_mod
        else:
            h1e_mod = numpy.zeros(self.H1.shape, dtype=self.H1.dtype)
            construct_h1e_mod(chol, self.H1, h1e_mod)
            self.h1e_mod = h1e_mod
        self.ktwist = numpy.array([None])
        # For consistency
        self.vol = 1.0
        self.hs_pot = self.chol_vecs


        u, s, vt = numpy.linalg.svd(self.hs_pot)

        self.nfields = self.nchol
        if verbose:
            print("# Number of Cholesky vectors: %d"%(self.nchol))
            print("# Number of fields: %d"%(self.nfields))
        if write_ints:
            self.write_integrals()
        if verbose:
            print("# Finished setting up Generic system object.")

    def hijkl(self, i, j, k, l):
        ik = i*self.nbasis + k
        jl = j*self.nbasis + l
        return numpy.dot(self.chol_vecs[ik,:], self.chol_vecs[jl,:])

    def write_integrals(self, filename='hamil.h5'):
        if self.sparse:
            write_qmcpack_sparse(self.H1[0],
                                 self.chol_vecs.reshape((-1,self.nbasis*self.nbasis)).T.copy(),
                                 self.nelec, self.nbasis,
                                 ecuc=self.ecore, filename=filename)
        else:
            write_qmcpack_dense(self.H1[0],
                                self.chol_vecs,
                                self.nelec, self.nbasis,
                                enuc=self.ecore, filename=filename,
                                real_chol=not self.cplx_chol)

def read_integrals(integral_file):
    try:
        (h1e, schol_vecs, ecore, nbasis, nup, ndown) = (
                from_qmcpack_sparse(integral_file)
                )
        chol_vecs = schol_vecs.toarray()
    except KeyError:
        (h1e, chol_vecs, ecore, nbasis, nup, ndown) = (
                from_qmcpack_dense(integral_file)
                )
    except OSError:
        print("# Unknown Hamiltonian file {}.".format(integral_file))
    except:
        print("# Unknown Hamiltonian file format.")
    return h1e, chol_vecs, ecore

def construct_h1e_mod(chol, h1e, h1e_mod):
    # Subtract one-body bit following reordering of 2-body operators.
    # Eqn (17) of [Motta17]_
    nbasis = h1e.shape[-1]
    chol_3 = chol.reshape((nbasis, nbasis, -1))
    # assert chol_3.__array_interface__['data'][0] == chol.__array_interface__['data'][0]
    v0 = 0.5 * numpy.einsum('ikn,jkn->ij', chol_3, chol_3, optimize='optimal')
    h1e_mod[0,:,:] = h1e[0] - v0
    h1e_mod[1,:,:] = h1e[1] - v0
