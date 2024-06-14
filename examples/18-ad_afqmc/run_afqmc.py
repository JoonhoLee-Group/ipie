import torch
import h5py
import numpy as np
from pyscf import gto, scf
from functools import partial
from torch.func import jvp

from ipie.qmc.comm import FakeComm
from ipie.addons.adafqmc.utils.miscellaneous import generate_hamiltonian_from_pyscf, get_hf_wgradient, trial_tangent
from ipie.addons.adafqmc.qmc.adafqmc import ADAFQMC

torch.set_printoptions(10)

comm = FakeComm()

atomstring = f'C 0 -1.24942055 0; O 0 0.89266692 0'
mol = gto.M(atom=atomstring, basis='cc-pvdz', verbose=3, symmetry=0, unit='bohr')
mf = scf.RHF(mol)
mf.kernel()

print("# nao = %d, nelec = %d" % (mol.nao_nr(), mol.nelec[0]))
#dipole moment calculation
num_frozen = 2

basis_change_matrix = torch.tensor(mf.mo_coeff, dtype=torch.float64)
# ao_dip
ao_dip = -mol.intor_symmetric('int1e_r', comp=3)
ao_dip = torch.tensor(ao_dip, dtype=torch.float64)

# mo_dip
mo_dip = torch.zeros_like(ao_dip, dtype=torch.float64)
for i in range(ao_dip.shape[0]):
    mo_dip[i] = basis_change_matrix.conj().t() @ ao_dip[i] @ basis_change_matrix

nuc_dip = np.zeros(3)
# nuc_dip
for i in range(mol.natm):
    for j in range(3):
        nuc_dip[j] += mol.atom_charge(i) * mol.atom_coord(i)[j]

# effective frozen operator
mo_dip_act = torch.zeros((ao_dip.shape[0], mol.nao - num_frozen, mol.nao - num_frozen), dtype=torch.float64)
for i in range(ao_dip.shape[0]):
    mo_dip_act[i] = mo_dip[i][num_frozen:, num_frozen:]
    nuc_dip[i] += 2. * torch.trace(mo_dip[i][:num_frozen, :num_frozen])

dip = (mo_dip_act[1], torch.tensor([nuc_dip[1]]))

hamobs = generate_hamiltonian_from_pyscf(mf, num_frozen=num_frozen, ortho_ao=False, observable=dip, obs_type='dipole')
print(f"# shape of cholesky vector: {hamobs.chol.shape}")

# compute trial and the tangent of trial
ovlp = mol.intor_symmetric('int1e_ovlp')
overlap = torch.tensor(ovlp, dtype=torch.float64)
orthtrial, _ = torch.linalg.qr(basis_change_matrix[:, num_frozen:].t() @ overlap @ basis_change_matrix[:, num_frozen:])

coupling = torch.tensor([0.], dtype=torch.float64, requires_grad=True)

partial_trial = partial(get_hf_wgradient, orthtrial, mol.nelec[0] - num_frozen, torch.eye(mol.nao_nr() - num_frozen, dtype=torch.float64), hamobs, 'RHF')
trial_nondetached, grad = jvp(partial_trial, (coupling, ), (torch.tensor([1.], dtype=torch.float64),))
trial_detached = trial_nondetached.detach().clone()
tg = grad.detach().clone()

# These parameters are used for testing purpose. It is not a production-level calculation.
options = {
    "num_steps_per_block": 10,
    "timestep": 0.01,
    "ad_block_size": 10,
    "num_walkers_per_process": 10,
    "num_ad_blocks": 10,
    "pop_control_freq": 5,
    "stabilize_freq": 5,
    "grad_checkpointing": False,
    "chkpt_size" : 50
}


adafqmc = ADAFQMC.build(comm, trial_tangent, **options)
energy, obs = adafqmc.run(hamobs, trial_detached, tg)