"""Utilities for runnning dice through the shciscf plugin."""

import glob
import os
import struct
from typing import Tuple

import h5py
import numpy as np
from pyscf import scf  # pylint: disable=import-error

try:
    from pyscf.shciscf import shci  # pylint: disable=import-error
except (ModuleNotFoundError, ImportError):
    print("issues importing pyscf.shciscf")
    raise ImportError

from ipie.qmc.afqmc import AFQMC
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.utils.from_pyscf import generate_hamiltonian


def get_perm(from_orb: list, to_orb: list, di: list, dj: list) -> int:
    """Determine sign of permutation needed to align two determinants."""
    nmove = 0
    perm = 0
    for o in from_orb:
        io = np.where(dj == o)[0]
        perm += io - nmove
        nmove += 1
    nmove = 0
    for o in to_orb:
        io = np.where(di == o)[0]
        perm += io - nmove
        nmove += 1
    return perm % 2 == 1


def read_dice_wavefunction(filename):
    print(f"Reading Dice wavefunction from {filename}")
    with open(filename, "rb") as f:
        data = f.read()
    _chr = 1
    _int = 4
    _dou = 8
    ndets_in_file = struct.unpack("<I", data[:4])[0]
    norbs = struct.unpack("<I", data[4:8])[0]
    wfn_data = data[8:]
    coeffs = []
    occs = []
    start = 0
    print(f"Number of determinants in dets.bin : {ndets_in_file}")
    print(f"Number of orbitals : {norbs}")
    for _ in range(ndets_in_file):
        coeff = struct.unpack("<d", wfn_data[start : start + _dou])[0]
        coeffs.append(coeff)
        start += _dou
        occ_i = wfn_data[start : start + norbs]
        occ_lists = decode_dice_det(str(occ_i)[2:-1])
        occs.append(occ_lists)
        start += norbs
    print("Finished reading wavefunction from file.")
    oa, ob = zip(*occs)
    return (
        np.array(coeffs, dtype=np.complex128),
        np.array(oa, dtype=np.int32),
        np.array(ob, dtype=np.int32),
    )


def convert_phase(coeff0, occa_ref, occb_ref, verbose=False):
    print("Converting phase to account for abab -> aabb")
    ndets = len(coeff0)
    coeffs = np.zeros_like(coeff0)
    for i in range(ndets):
        if verbose and i % max(1, (int(0.1 * ndets))) == 0 and i > 0:
            done = float(i) / ndets
            print(f"convert phase {i}. Percent: {done}")
        # doubles = list(set(occa_ref[i]) & set(occb_ref[i]))
        occa0 = np.array(occa_ref[i])
        occb0 = np.array(occb_ref[i])

        count = 0
        for ocb in occb0:
            passing_alpha = np.where(occa0 > ocb)[0]
            count += len(passing_alpha)

        phase = (-1) ** count
        coeffs[i] = coeff0[i] * phase
    ixs = np.argsort(np.abs(coeffs))[::-1]
    coeffs = coeffs[ixs]
    occa = np.array(occa_ref)[ixs]
    occb = np.array(occb_ref)[ixs]

    return coeffs, occa, occb


def decode_dice_det(occs):
    occ_a = []
    occ_b = []
    for i, occ in enumerate(occs):
        if occ == "2":
            occ_a.append(i)
            occ_b.append(i)
        elif occ == "a":
            occ_a.append(i)
        elif occ == "b":
            occ_b.append(i)
    return occ_a, occ_b


def read_opdm(filename: str) -> np.ndarray:
    """Read DICE RDM from filename."""
    with open(filename) as f:
        data = f.readlines()
        nmo = int(data[0])
        opdm = np.zeros((nmo, nmo))
        for l in data[1:]:
            _l = l.split()
            i, j, v = int(_l[0]), int(_l[1]), float(_l[2])
            opdm[i, j] = v
            opdm[j, i] = v

    return opdm


def get_noons(opdm: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    eig_val, eig_vec = np.linalg.eigh(opdm)
    # Want natural ordering (i.e. high to low) for noons
    return eig_val[::-1], eig_vec[:, ::-1]


def get_active_space_from_noons(
    noons: np.ndarray,
    nelec: tuple,
    thresh: float = 0.02,
    lower_bound_scaling_factor: int = 1,
) -> Tuple[int, Tuple[int, int]]:
    """Attempt to determine active space from SHCI noons."""
    frozen = noons >= 2.0 - thresh
    inactive = noons < thresh
    active_orbs = np.where(~frozen & ~inactive)[0]
    print(f"frozen occupancies: {noons[frozen]}")
    print(f"active occupancies: {noons[active_orbs]}")
    # find orbitals that likely correspond to electrons in active space
    # reference wavefunction.
    # lower_bound_scaling_factor accounts for picking singly occupied orbitals.
    occ_in_active_space = np.where(
        (noons <= 2.0 - thresh) & (noons > 1 - lower_bound_scaling_factor * thresh)
    )[0]
    nalpha, nbeta = nelec
    frozen_fermi_level = occ_in_active_space[0]
    nalpha_active = nalpha - frozen_fermi_level
    nbeta_active = nbeta - frozen_fermi_level
    num_occ_orbs = len(occ_in_active_space)
    expected_occ_orbs = max(nalpha_active, nbeta_active)
    err_string = f"Consider changing value of lower_bound_scaling_factor: {num_occ_orbs} vs {expected_occ_orbs}"
    assert num_occ_orbs == expected_occ_orbs, err_string
    return (len(active_orbs), (nalpha_active, nbeta_active))


def run_shci_coarse(
    mf: scf.RHF,
    cas: Tuple[int, Tuple[int, int]],
    num_proc: int,
    eps=1e-4,
    num_sweep=3,
    mpi_cmd="mpirun -np",
    num_dets=100000,
) -> shci.SHCI:
    """Run coarse SHCI calculation in presumably large active space.

    Typically we run this first and obtain the SHCI 1RDM, from which one can
    compute NOONS and hopefully select a useful active space.
    """
    assert eps < 1e-3
    assert num_sweep < 5
    sweep_epsilon = np.logspace(-3, np.log10(eps), num_sweep)
    sweep_iter = np.linspace(0, 5, num_sweep, dtype=int)
    nact, nelec = cas
    mc1 = shci.SHCISCF(mf, nact, nelec)
    print(f"Coarse SHCI: nact = {mc1.ncas}, ncore = {mc1.ncore}")
    path = os.getcwd()
    mc1.fcisolver.scratchDirectory = path
    mc1.fcisolver.runtimeDir = path
    mc1.chkfile = "mcscf.chk"
    mc1.fcisolver.sweep_iter = sweep_iter
    mc1.fcisolver.sweep_epsilon = sweep_epsilon
    mc1.fcisolver.nPTiter = 0
    mc1.max_iter = 1
    mc1.fcisolver.maxIter = 20
    mc1.max_cycle_macro = 1
    mc1.internal_rotation = True
    mc1.fcisolver.nPTiter = 0  # Turns off PT calculation, i.e. no PTRDM.
    mc1.fcisolver.mpiprefix = mpi_cmd + f" {num_proc}"
    shci.dryrun(mc1)
    shci.writeSHCIConfFile(mc1.fcisolver, mc1.nelecas, False)
    with open(path + "/" + mc1.fcisolver.configFile, "a") as f:
        f.write("noio\n")
        f.write(f"writebestdeterminants {num_dets}\n\n")
    shci.executeSHCI(mc1.fcisolver)
    mc1.e_tot = shci.readEnergy(mc1.fcisolver)
    print(f"Coarse SHCI total energy = {mc1.e_tot}")
    return mc1


def run_shciscf(
    mf: scf.RHF,
    cas: Tuple[int, Tuple[int, int]],
    nat_orbs: np.ndarray,
    num_proc: int,
    eps=1e-5,
    num_sweep=3,
    mpi_cmd="mpirun -np",
    num_dets=1000000,
):
    """Helper function to run SHCISCF and write determinants to a format useable
    by ipie.
    """
    assert eps < 1e-3
    assert num_sweep < 5
    scf.chkfile.dump_scf(mf.mol, "mcscf.chk", mf.e_tot, mf.mo_energy, mf.mo_coeff, mf.mo_occ)
    mc1 = shci.SHCISCF(mf, cas[0], cas[1])
    print(f"SHCISCF: nact = {mc1.ncas}, ncore = {mc1.ncore}")
    path = os.getcwd()
    sweep_epsilon = np.logspace(-3, np.log10(eps), num_sweep)
    sweep_iter = np.linspace(0, 5, num_sweep, dtype=int)
    mc1.fcisolver.scratchDirectory = path
    mc1.fcisolver.runtimeDir = path
    mc1.chkfile = mf.chkfile
    mc1.fcisolver.sweep_iter = sweep_iter
    mc1.fcisolver.sweep_epsilon = sweep_epsilon
    mc1.max_iter = 20
    mc1.fcisolver.maxIter = 20
    mc1.max_cycle_macro = 20
    mc1.internal_rotation = True
    mc1.fcisolver.nPTiter = 100
    mc1.fcisolver.mpiprefix = mpi_cmd + f" {num_proc}"
    mc1.chkfile = mc1._scf.chkfile
    mc1.kernel(nat_orbs)
    shci.dryrun(mc1)
    shci.writeSHCIConfFile(mc1.fcisolver, mc1.nelecas, False)
    with open(path + "/" + mc1.fcisolver.configFile, "a") as f:
        f.write("noio\n")
        f.write(f"writebestdeterminants {num_dets}\n\n")
    shci.executeSHCI(mc1.fcisolver)
    bkp_files = glob.glob("*.bkp")
    for bkp_file in bkp_files:
        os.remove(bkp_file)
    mc1.e_tot = shci.readEnergy(mc1.fcisolver)
    print(f"Coarse SHCI total energy = {mc1.e_tot}")
    return mc1


def build_trial_from_shciscf(
    mf: scf.RHF,
    nmo: int,
    nelec: Tuple[int, int],
    num_procs: int,
    noons_fname: str = "noons.h5",
    noons_thresh: float = 0.02,
    num_dets_for_trial: int = 1000000,
    convert_det_phase: bool = False,
) -> Tuple[ParticleHole, shci.SHCI]:
    """Wrapper function to generate trial using SHCI to select the active space.

    No guarantees about if this works well or is a sensible thing to do. Users
    should inspect noons etc.
    """
    print(f"Running coarse SHCI calculation in active space: ({nelec}, {nmo})")
    shci_inst = run_shci_coarse(mf, (nmo, nelec), num_procs)
    opdm = read_opdm(f"spatial1RDM.0.0.txt")
    print(f"Determining active space using 1RDM in")
    noons, nat_orbs = get_noons(opdm)
    print(f"Writing noons to {noons_fname}")
    with h5py.File(f"{noons_fname}", "w") as fh5:
        fh5["noons"] = noons
        fh5["nat_orb"] = nat_orbs
    cas = get_active_space_from_noons(
        noons,
        nelec,
        thresh=noons_thresh,
    )
    print("active space determined from coarse SHCI noons: ", cas)
    print(f"Natural orbital unitary shape: {nat_orbs.shape}")
    C = np.copy(mf.mo_coeff)
    # Rotate MOs by natural orbitals in active space selected.
    ncore = shci_inst.ncore
    nact = shci_inst.ncas
    print(f"Number of core orbitals in coarse SHCI: {ncore}")
    print(f"Number of active orbitals in coarse SHCI: {nact}")
    print(
        f"Coarse SCHI active space slice in original full set of orbitals: [{ncore}:{ncore+nact}]."
    )
    Cfrzn = C[:, :ncore].copy()
    Cactv = np.dot(C[:, ncore : ncore + nact], nat_orbs)
    Cvirt = C[:, ncore + nact :].copy()
    init_guess = np.hstack([Cfrzn, Cactv, Cvirt])
    shci_inst = run_shciscf(mf, cas, init_guess, num_procs, eps=1e-5)
    wfn = read_dice_wavefunction("dets.bin")
    if convert_det_phase:
        wfn = convert_phase(*wfn)
    nelec = mf.mol.nelec
    nmo = mf.mo_coeff.shape[-1]
    trial = ParticleHole(wfn, nelec, nmo, num_dets_for_trial=num_dets_for_trial)
    return trial, shci_inst


def build_driver_from_shciscf(
    mf: scf.RHF,
    cas: Tuple[int, Tuple[int, int]],
    num_procs: int,
    num_dets_for_trial: int = 1000000,
    num_walkers: int = 100,
    noons_thresh: float = 0.02,
    convert_det_phase: bool = False,
    chol_cut=1e-8,
    seed=7,
) -> "AFQMC":
    nact, nelec_cas = cas
    trial, shci_inst = build_trial_from_shciscf(
        mf,
        nact,
        nelec_cas,
        num_procs,
        num_dets_for_trial=num_dets_for_trial,
        noons_thresh=noons_thresh,
        convert_det_phase=convert_det_phase,
    )

    ham = generate_hamiltonian(
        mf.mol, shci_inst.mo_coeff, mf.get_hcore(), shci_inst.mo_coeff, chol_cut=chol_cut
    )

    afqmc = AFQMC.build(mf.mol.nelec, ham, trial)
    return afqmc, shci_inst
