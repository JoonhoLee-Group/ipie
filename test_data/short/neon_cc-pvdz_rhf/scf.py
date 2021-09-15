from pyscf import gto, scf, cc

atom = gto.M(atom=[('H', 1.6*i, 0, 0) for i in range(0,10)],
             basis='sto-6g',
             verbose=4,
             unit='Bohr')
mf = scf.UHF(atom)
mf.chkfile = 'scf.chk'
mf.kernel()
