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

from ipie.addons.free_projection.qmc.calc import build_fpafqmc_driver

qmc_options = {
    "num_iterations_fp": 100,
    "num_blocks": 5,
    "num_steps": 20,
    "num_walkers": 10,
    "dt": 0.05,
}
afqmc = build_fpafqmc_driver(
    comm,
    nelec=mol.nelec,
    seed=41100801,
    qmc_options=qmc_options,
)
if comm.rank == 0:
    print(afqmc.params)  # Inspect the default qmc options
afqmc.run()

# analysis
if comm.rank == 0:
    from ipie.addons.free_projection.analysis.extraction import extract_observable
    from ipie.addons.free_projection.analysis.jackknife import jackknife_ratios

    for i in range(afqmc.params.num_blocks):
        print(
            f"\nEnergy statistics at time {(i+1) * afqmc.params.num_steps_per_block * afqmc.params.timestep}:"
        )
        qmc_data = extract_observable(afqmc.estimators[i].filename, "energy")
        energy_mean, energy_err = jackknife_ratios(qmc_data["ENumer"], qmc_data["EDenom"])
        print(f"Energy: {energy_mean:.8e} +/- {energy_err:.8e}")
