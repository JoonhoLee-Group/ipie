import numpy as np
import os
import pytest

from ipie.utils.io import write_hamiltonian, read_hamiltonian

@pytest.mark.unit
def test_read_write():
    nmo = 10
    naux = 100
    hcore = np.random.random((nmo, nmo))
    LXmn = np.random.random((naux, nmo, nmo))
    e0 = 18.0
    write_hamiltonian(hcore, LXmn, e0, filename='test.h5')
    hcore_read, LXmn_read, e0_read = read_hamiltonian('test.h5')
    assert np.allclose(hcore_read, hcore)
    assert np.allclose(LXmn_read, LXmn)
    assert e0 == pytest.approx(e0_read)

def teardown_module(self):
    cwd = os.getcwd()
    files = ['test.h5']
    for f in files:
        try:
            os.remove(cwd + "/" + f)
        except OSError:
            pass
