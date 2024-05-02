from pyscf import cc, gto, scf
from ipie.utils.mpi import make_splits_displacements
import h5py
import numpy as np
import gc 


mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 4)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)
mf = scf.UHF(mol)
mf.chkfile = "scf.chk"
mf.kernel()

from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk
gen_ipie_input_from_pyscf_chk(mf.chkfile, verbose=0)


from ipie.utils.chunk_large_chol import split_cholesky
split_cholesky('hamiltonian.h5', 4) # split the cholesky to 4 subfiles