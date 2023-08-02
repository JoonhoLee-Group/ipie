import numpy as np
from pyscf import cc, gto, scf

from ipie.config import MPI
from ipie.utils.from_pyscf import gen_ipie_input_from_pyscf_chk

comm = MPI.COMM_WORLD
mol = gto.M(
    atom=[("H", 1.6 * i, 0, 0) for i in range(0, 10)],
    basis="sto-6g",
    verbose=4,
    unit="Bohr",
)
if comm.rank == 0:
    mf = scf.UHF(mol)
    mf.chkfile = "scf.chk"
    mf.kernel()
    mycc = cc.UCCSD(mf).run()
    et = mycc.ccsd_t()
    print("UCCSD(T) energy {}".format(mf.e_tot + mycc.e_corr + et))

    gen_ipie_input_from_pyscf_chk(mf.chkfile, verbose=0)
comm.barrier()

from ipie.qmc.calc import build_afqmc_driver

# fixing random seed for reproducibility
afqmc = build_afqmc_driver(comm, nelec=mol.nelec, num_walkers_per_task=100, seed=41100801)
if comm.rank == 0:
    print(afqmc.qmc)  # Inspect the default qmc options

# Let us override the number of blocks to keep it short
afqmc.qmc.nblocks = 10
afqmc.estimators.overwite = True
afqmc.run(comm=comm)

if comm.rank == 0:
    # We can extract the qmc data as as a pandas data frame like so
    from ipie.analysis.extraction import extract_observable

    qmc_data = extract_observable(afqmc.estimators.filename, "energy")
    y = qmc_data["ETotal"]
    y = y[1:]  # discard first 50 blocks

    from ipie.analysis.autocorr import reblock_by_autocorr

    df = reblock_by_autocorr(y, verbose=1)

    # assert np.isclose(df.at[0,'ETotal_ac'], -5.325611614468466)
    # assert np.isclose(df.at[0,'ETotal_error_ac'], 0.00938082351500978)
