import numpy
import h5py
import scipy.sparse
from pyscf import gto, scf, mcscf, fci, ao2mo, lib
from pauxy.systems.generic import Generic
from pauxy.utils.from_pyscf import generate_integrals
from pauxy.utils.io import (
        write_qmcpack_wfn,
        write_qmcpack_dense,
        write_input
        )

mol = gto.M(atom=[('N', 0, 0, 0), ('N', (0,0,3.0))], basis='sto-3g', verbose=3,
            unit='Bohr')
mf = scf.RHF(mol)
mf.chkfile = 'scf.chk'
ehf = mf.kernel()
M = 6
N = 6
mc = mcscf.CASSCF(mf, M, N)
mc.chkfile = 'scf.chk'
mc.kernel()
e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
print(ehf, e_tot)
# Rotate by casscf mo coeffs.
h1e, chol, nelec, enuc = generate_integrals(mol, mf.get_hcore(), mo,
                                            chol_cut=1e-5, verbose=True)
write_qmcpack_dense(h1e, chol.T.copy(), nelec,
                    h1e.shape[-1], enuc=enuc, filename='afqmc.h5')
coeff, occa, occb = zip(*fci.addons.large_ci(fcivec, M, (3,3),
                                         tol=0.1, return_strs=False))
core = [i for i in range(mc.ncore)]
occa = [numpy.array(core + [o + mc.ncore for o in oa]) for oa in occa]
occb = [numpy.array(core + [o + mc.ncore for o in ob]) for ob in occb]
coeff = numpy.array(coeff,dtype=numpy.complex128)
# Sort in ascending order.
ixs = numpy.argsort(numpy.abs(coeff))[::-1]
coeff = coeff[ixs]
occa = numpy.array(occa)[ixs]
occb = numpy.array(occb)[ixs]

nmo = mf.mo_coeff.shape[-1]
rdm = mc.make_rdm1()
eigs, eigv = numpy.linalg.eigh(rdm)
psi0a = eigv[::-1,:mol.nelec[0]].copy()
psi0b = eigv[::-1,:mol.nelec[1]].copy()
psi0 = [psi0a, psi0b]
write_qmcpack_wfn('afqmc.h5', (coeff,occa,occb), 'uhf',
                  mol.nelec, nmo, init=psi0, mode='a')
write_input('input.json', 'afqmc.h5', 'afqmc.h5')
