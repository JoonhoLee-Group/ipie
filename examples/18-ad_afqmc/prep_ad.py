import torch
import h5py
import numpy as np
from pyscf import gto, scf, lo
from ipie.addons.adafqmc.utils.miscellaneous import generate_hamiltonian_from_pyscf, get_hf_wgradient, dump_hamiltonian
from functools import partial
from torch.func import jvp

torch.set_printoptions(10)

atomstring = f'C 0 -1.24942055 0; O 0 0.89266692 0'
mol = gto.M(atom=atomstring, basis='cc-pvdz', verbose=3, symmetry=0, unit='bohr')
mf = scf.RHF(mol)
mf.kernel()

print("# nao = %d, nelec = %d" % (mol.nao_nr(), mol.nelec[0]))
print("# calculating dipole moment")
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

print("# finished calculating dipole")

print("# start to generate hamiltonian")
hamobs = generate_hamiltonian_from_pyscf(mf, num_frozen=num_frozen, ortho_ao=False, observable=dip, obs_type='dipole')
print(f"# shape of cholesky vector: {hamobs.chol.shape}")
dump_hamiltonian(hamobs, 'my_hamiltonian.h5')
print("# finished generating hamiltonian")

# compute trial and the tangent of trial
ovlp = mol.intor_symmetric('int1e_ovlp')
overlap = torch.tensor(ovlp, dtype=torch.float64)
orthtrial, _ = torch.linalg.qr(basis_change_matrix[:, num_frozen:].t() @ overlap @ basis_change_matrix[:, num_frozen:])

coupling = torch.tensor([0.], dtype=torch.float64, requires_grad=True)

partial_trial = partial(get_hf_wgradient, orthtrial, mol.nelec[0] - num_frozen, torch.eye(mol.nao_nr() - num_frozen, dtype=torch.float64), hamobs, 'RHF')
print("# start to compute the gradient of trial")
trial_nondetached, grad = jvp(partial_trial, (coupling, ), (torch.tensor([1.], dtype=torch.float64),))
trial_detached = trial_nondetached.detach().clone()
print("# finished computing the gradient of trial")
tg = grad.detach().clone()
print(f"# trial_detached: {trial_detached}")
print(f"# gradient maximum element: {torch.max(torch.abs(tg))}, gradient minimum element: {torch.min(torch.abs(tg))}")

with h5py.File('trial_with_tangent.h5', 'w') as fL:
    fL.create_dataset('trial', data=trial_detached)
    fL.create_dataset('tangent', data=tg)
    fL.create_dataset('mocoeff', data=mf.mo_coeff)