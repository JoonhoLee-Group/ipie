"""usage: python run_afqmc.py

One semi-blackbox way of constructing an active space is to use NOONs computed
from some cheaper approximate wavefunction method (UHF, MP2, ...). Here we use
SHCI which has the advantage of being able to simulate large active spaces at
least for relatively coarse accuracy.

The procedure is:

1. Run a coarse SHCI calculation in a large active space for our system.
2. Compute the SHCI one-rdm and resultant noons / natural orbitals.
3. Pick an active space based on some noon threshold.
4. Rotate our orbitals by the unitary matrix specifying these natural orbitals.
5. Rerun a tight (i.e. accurate) SHCISCF calculation in the (hopefully) smaller
active space we determined from step 3. Here we use Dice as it can treat much
larger active spaces than traditional approaches.
6. Use this trial wavefunction (which is very accurate in this active space) as
a trial wavefunction for AFQMC in the **full** space of our original problem,
i.e. we reinsert any frozen core orbitals or discarded virtuals. The trial
wavefunction captures any static correlation while afqmc should mop up the
missing dynamic correlation.

These 6 steps are folded into the utility factory method
build_driver_from_shciscf which returns the appropriately constructed AFQMC
driver.

Note active space selection is challenging and there is no guarantee this
procedure will always work OR that the automations we provide are always
appropriate. Check out ipie.utils.from_dice for the individual functions which
implement the workflow. In particular, the defaults (which may not be
controllable from the factory method) may require tuning. One should always
visualize the active space too (and the noons) to make sure things are sensible.
"""
import numpy as np

try:
    from pyscf import gto, scf

    from ipie.utils.from_dice import build_driver_from_shciscf
except ImportError:
    import sys

    print("pyscf, dice, and shciscf plugin are needed for this example.")
    sys.exit(0)


mol = gto.Mole()
mol.basis = "cc-pvdz"
mol.atom = """N 0 0 0
N 0 0 2.0
"""
mol.verbose = 4
mol.build()
mol.symmetry = False

mf = scf.RHF(mol)
mf.kernel()

nmo = mf.mo_coeff.shape[-1]
nelec = mf.mol.nelec
# Freeze two orbitals
cas = (mf.mo_coeff.shape[-1] - 2, (nelec[0] - 2, nelec[0] - 2))

# dice requires MPI, num_proc specifies how many MPI processes you want to run
num_proc = 8
afqmc, shciscf_inst = build_driver_from_shciscf(
    mf,
    cas,
    num_proc,
    noons_thresh=0.05,  # threshold for including natural orbital in our active space (i.e. occupancies of 1.95 - 0.05 are included)
    convert_det_phase=False,  # This is apparenty not needed for dice?
    chol_cut=1e-8,  # note the agreement between the trial energy and shciscf is controlled by this parameter.
)
ham = afqmc.hamiltonian
sys = afqmc.system
# ALWAYS check the trial wavefunction energy is consistent.
afqmc.trial.calculate_energy(sys, ham)
assert np.isclose(afqmc.trial.energy.real, shciscf_inst.e_tot, atol=1e-8)
# ... run QMC etc
