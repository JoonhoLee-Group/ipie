import h5py
import numpy
import scipy.sparse
from pyscf import ao2mo, fci, gto, lib, mcscf, scf

from ipie.systems.generic import Generic
from ipie.utils.from_pyscf import generate_integrals
from ipie.utils.io import write_input, write_qmcpack_dense, write_qmcpack_wfn

nocca = 4
noccb = 2

mol = gto.M(
    atom=[("N", 0, 0, 0), ("N", (0, 0, 3.0))],
    basis="ccpvdz",
    verbose=3,
    spin=nocca - noccb,
    unit="Bohr",
)
mf = scf.RHF(mol)
mf.chkfile = "scf.chk"
ehf = mf.kernel()
M = 6
N = 6
mc = mcscf.CASSCF(mf, M, N)
mc.chkfile = "scf.chk"
mc.kernel()
e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
print(ehf, e_tot)
# Rotate by casscf mo coeffs.
h1e, chol, nelec, enuc, cas_idx = generate_integrals(
    mol, mf.get_hcore(), mo, chol_cut=1e-5, verbose=True
)
write_qmcpack_dense(
    h1e, chol.T.copy(), nelec, h1e.shape[-1], enuc=enuc, filename="afqmc.h5"
)
coeff, occa, occb = zip(
    *fci.addons.large_ci(fcivec, M, (nocca, noccb), tol=0.1, return_strs=False)
)
core = [i for i in range(mc.ncore)]
occa = [numpy.array(core + [o + mc.ncore for o in oa]) for oa in occa]
occb = [numpy.array(core + [o + mc.ncore for o in ob]) for ob in occb]
coeff = numpy.array(coeff, dtype=numpy.complex128)
# Sort in ascending order.
ixs = numpy.argsort(numpy.abs(coeff))[::-1]
coeff = coeff[ixs]
occa = numpy.array(occa)[ixs]
occb = numpy.array(occb)[ixs]

nmo = mf.mo_coeff.shape[-1]
rdm = mc.make_rdm1()
eigs, eigv = numpy.linalg.eigh(rdm)
psi0a = eigv[::-1, occa[0]].copy()
psi0b = eigv[::-1, occb[0]].copy()

nocca = mol.nelec[0]
noccb = mol.nelec[1]
nbsf = psi0a.shape[0]

psi0a += 1e-1 * numpy.random.randn(nocca * nbsf).reshape(nbsf, nocca)
psi0b += 1e-1 * numpy.random.randn(noccb * nbsf).reshape(nbsf, noccb)

psi0 = [psi0a, psi0b]
write_qmcpack_wfn(
    "afqmc.h5", (coeff, occa, occb), "uhf", mol.nelec, nmo, init=psi0, mode="a"
)
# write_input('input.json', 'afqmc.h5', 'afqmc.h5')
