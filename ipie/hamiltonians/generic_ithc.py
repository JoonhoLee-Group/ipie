import numpy as np
import cmath
import h5py
from ipie.utils.backend import cast_to_device
from ipie.hamiltonians.generic_base import GenericBase


def conjmat(M): #conjugates a matrix
    return np.conjugate(M.T)



def import_Hamiltonian(path_to_hamiltonian,path_to_isometry):
    """Imports the one-electron, two-electron parts and an Isometry U to transform
     into an extended basis"""

    with h5py.File(path_to_hamiltonian, "r") as f:
    # List all groups
        print("Keys:", list(f.keys()))
        V_nuc=f['enuc']*np.eye(2)
        V_e=np.asarray(f['eri'])
        kin_e=np.asarray(f['hkin'])
        kin_nuc=np.asarray(f['hnuc'])
        # state= numpy.asarray(f['phi']) #real space representation of two orbitals on a grid 
    f.close()
    isometry = np.asarray(np.load(path_to_isometry))
    return V_nuc, V_e, kin_e, kin_nuc, isometry

def eri_diag(isometry,eri, tol=1e-13):
        no,m = isometry.shape #shape 2x3
        rcond = tol
        MV = eri.reshape(no*no,no*no) #reshape of eri 4x4
        B = np.einsum('ix,jx->ijx', isometry, isometry) 
        MB = B.reshape(no**2,m) #4x3
        Binv = np.linalg.pinv(MB,rcond=rcond) #inverse of B 3x4
        W = Binv@MV@conjmat(Binv) #diagonal eri 3x3
        #T = Binv@(kin_e.flatten()) #kinetic terms 
        #P = Binv@(kin_nuc.flatten()) 
        errV = np.linalg.norm(MB @ W @ conjmat(MB) - MV)/np.linalg.norm(MV)
        #print("Error eri=",errV)
        return W #,T,P #change this to input


def mat_decomp(A, tol=None):
    """
    For real symmetric A, returns C (possibly complex) such that:
        A ≈ C.T @ C     (NOTE: transpose, not conjugate-transpose)
    If truncate_zeros=True, drops (near-)zero eigenvalues/columns.
    """
    A = np.asarray(A)
    w, Q = np.linalg.eigh(A)

    if tol is None:
        tol = (np.max(np.abs(w)) * 1e-12) if w.size else 0.0

    keep = np.abs(w) > tol
    w, Q = w[keep], Q[:, keep]
    
    sqrt_w = np.sqrt(w.astype(np.complex128))   # negative -> imaginary
    C = (Q * sqrt_w).T                              # scale columns

    return C

class GenericITHC(GenericBase):
    """
    Class for Hamiltonian in extended basis. 
    T: one body terms
    W: two body terms in the extended basis
    Isometry: Transformation into extended basis set
    n_old: Size of original space
    n_new:Size of extended space
    """

    def __init__(self, h1e, isometry,W, ecore=0.0,verbose = False ): #needs the one body and eri as an input
        
        self.verbose = verbose
        self.ecore = ecore
        self.nbasis = np.shape(isometry)[0]
        self.nbasis_extended = np.shape(isometry)[1]

        self.isometry= np.asarray(isometry) #Isometry for diagonalisation
        # self.eri = eri
        self.W= np.asarray(W)
        self.H1= np.array([h1e,h1e])

        # self.eri = np.einsum("ia,ja,kb,lb,ab",Isometry,Isometry,Isometry,Isometry,W)

        W_ = np.tile(self.W, (2, 2)) 
        W_[np.diag_indices_from(W_)] = 0
        self.v = 1.j * mat_decomp(W_)


        #self.nbasis= self.old
        self.pairs = np.triu_indices(self.nbasis_extended, k=1) # only alpha < beta
        self.pairs = np.column_stack((self.pairs))
        self.nfields = len(self.pairs) #number of auxilliary fields
        
        if verbose:
            print("# Setting up one-body operator.")
        
        return 
    
    def isometry_test(self):
        test_matrix_l= self.isometry @ conjmat(self.isometry)
        test_matrix_r= conjmat(self.isometry) @ self.isometry 
        epsilon_l= np.linalg.norm(test_matrix_l - np.eye(self.nbasis))
        epsilon_r= np.linalg.norm(test_matrix_r - np.eye(self.nbasis_extended))
        try:
            assert epsilon_l < 1e-5 
        except:
            print(f"Isometry test failed with errors {epsilon_l}, {epsilon_r}")
        return 

    


    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose=verbose)





