from pyscf import gto, scf, cc

atom = gto.M(atom="P 0 0 0", basis="6-31G", verbose=4, spin=3, unit="Bohr")

mf = scf.UHF(atom)
mf.chkfile = "scf.chk"
mf.kernel()

mycc = mf.CCSD(frozen=list(range(5))).run()
et = mycc.ccsd_t()
print("UCCSD(T) energy", mycc.e_tot + et)
