import os
import pytest
try:
    from pyscf import gto, ao2mo, scf
    from pauxy.utils.from_pyscf import (
            integrals_from_scf,
            integrals_from_chkfile,
            get_pyscf_wfn,
            dump_pauxy
            )
    no_pyscf = False
except (ImportError, OSError):
    no_pyscf = True
from pauxy.utils.io import (
        write_input,
        read_qmcpack_wfn_hdf,
        from_qmcpack_sparse
        )

@pytest.mark.unit
@pytest.mark.skipif(no_pyscf, reason="pyscf not found.")
def test_from_pyscf():
    atom = gto.M(atom='Ne 0 0 0', basis='sto-3g', verbose=0, parse_arg=False)
    mf = scf.RHF(atom)
    mf.kernel()
    h1e, chol, nelec, enuc = integrals_from_scf(mf, verbose=0, chol_cut=1e-5)
    assert chol.shape[0] == 15
    assert chol.shape[1] == 25
    assert nelec == (5,5)
    assert h1e.shape[0] == 5

@pytest.mark.unit
@pytest.mark.skipif(no_pyscf, reason="pyscf not found.")
def test_from_chkfile():
    atom = gto.M(atom=[('H', 1.5*i, 0, 0) for i in range(0,10)],
                 basis='sto-6g', verbose=0, parse_arg=False)
    mf = scf.RHF(atom)
    mf.chkfile = 'scf.chk'
    mf.kernel()
    h1e, chol, nelec, enuc = integrals_from_chkfile('scf.chk', verbose=0, chol_cut=1e-5)
    assert h1e.shape == (10,10)
    assert chol.shape == (19,100)
    assert nelec == (5,5)
    assert enuc == pytest.approx(6.805106937254286)

@pytest.mark.unit
@pytest.mark.skipif(no_pyscf, reason="pyscf not found.")
def test_pyscf_to_pauxy():
    atom = gto.M(atom=[('H', 1.5*i, 0, 0) for i in range(0,4)],
                 basis='sto-6g', verbose=0, parse_arg=False)
    mf = scf.RHF(atom)
    mf.chkfile = 'scf.chk'
    mf.kernel()
    dump_pauxy(chkfile='scf.chk', hamil_file='afqmc.h5', sparse=True)
    wfn = read_qmcpack_wfn_hdf('afqmc.h5')
    h1e, chol, ecore, nmo, na, nb = from_qmcpack_sparse('afqmc.h5')
    write_input('input.json', 'afqmc.h5', 'afqmc.h5')

def teardown_module(self):
    cwd = os.getcwd()
    files = ['scf.chk', 'afqmc.h5', 'input.json']
    for f in files:
        try:
            os.remove(cwd+'/'+f)
        except OSError:
            pass
