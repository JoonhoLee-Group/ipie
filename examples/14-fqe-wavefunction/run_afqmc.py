# Copyright 2022 The ipie Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Fionn Malone <fmalone@google.com>
#
"""Convert an FQE wavefunction to ipie and vice-versa.

Play around with various thresholds to see how it affects the energy.
"""
import sys
from typing import List, Tuple, Union

try:
    import fqe
except (ImportError, ModuleNotFoundError):
    print("fqe required")
    sys.exit(0)
try:
    import pyscf
except (ImportError, ModuleNotFoundError):
    print("pyscf required")
    sys.exit(0)
import numpy as np
from pyscf import ao2mo, gto, mcscf, scf

from ipie.hamiltonians.generic import GenericRealChol as GenericHam
from ipie.systems.generic import Generic as GenericSys
from ipie.trial_wavefunction.particle_hole import ParticleHole
from ipie.utils.from_pyscf import generate_hamiltonian


def get_occa_occb_coeff_from_fqe_wfn(
    fqe_wf: fqe.Wavefunction, threshold: float = 0.0
) -> Tuple[List[np.ndarray], ...]:
    """Generate occlists from fqe wavefunction."""

    def _get_sector_data(sector, threshold, occa_list, occb_list, coeffs):
        for inda in range(sector._core.lena()):
            alpha_str = sector._core.string_alpha(inda)
            for indb in range(sector._core.lenb()):
                if np.abs(sector.coeff[inda, indb]) > threshold:
                    alpha_str = sector._core.string_alpha(inda)
                    beta_str = sector._core.string_beta(indb)
                    coeff = sector.coeff[inda, indb]

                    occa_list.append(fqe.bitstring.integer_index(alpha_str))
                    occb_list.append(fqe.bitstring.integer_index(beta_str))
                    coeffs.append(coeff)

    occa_list: List[np.ndarray] = []
    occb_list: List[np.ndarray] = []
    coeffs: List[np.ndarray] = []

    for sector_key in fqe_wf.sectors():
        sector = fqe_wf.sector(sector_key)
        _get_sector_data(sector, threshold, occa_list, occb_list, coeffs)

    return (np.asarray(coeffs), np.asarray(occa_list), np.asarray(occb_list))


def get_fqe_wfn_from_occ_coeff(
    coeffs: np.ndarray,
    occa: np.ndarray,
    occb: np.ndarray,
    n_elec: int,
    n_orb: int,
    ms: int = 0,
    threshold: float = 0.0,
) -> fqe.Wavefunction:
    """A helper function to map an AFQMC wavefunction to FQE.

    Args:
        coeffs: The ci coefficients
        occa: The alpha occupation strings.
        occb: The beta occupation strings.
        n_elec: Number of electrons.
        n_orb: number of orbitals.
        ms: spin polarization.
        threshold: ci coefficient threshold. A coefficient whose absolute value
            below this value is considered zero.
    """

    def _set_sector_data(sector, threshold, occa_list, occb_list, coeffs):
        fqe_graph = sector.get_fcigraph()
        for idet, (occa, occb) in enumerate(zip(occa_list, occb_list)):
            alpha_str = fqe.bitstring.reverse_integer_index(occa)
            beta_str = fqe.bitstring.reverse_integer_index(occb)
            inda = fqe_graph.index_alpha(alpha_str)
            indb = fqe_graph.index_alpha(beta_str)
            if np.abs(coeffs[idet]) > threshold:
                sector.coeff[inda, indb] = coeffs[idet]

    # ensure it is normalized
    _coeffs = coeffs / np.dot(coeffs.conj(), coeffs) ** 0.5
    fqe_wf = fqe.Wavefunction([[n_elec, ms, n_orb]])

    for sector_key in fqe_wf.sectors():
        sector = fqe_wf.sector(sector_key)
        _set_sector_data(sector, threshold, occa, occb, _coeffs)

    return fqe_wf


def get_fqe_variational_energy(
    ecore: float, h1e: np.ndarray, eris: np.ndarray, wfn: fqe.Wavefunction
) -> float:
    """Compute FQE variational energy from ERIs and FQE wavefunction."""
    # get integrals into openfermion order
    of_eris = np.transpose(eris, (0, 2, 3, 1))
    # ... and then into FQE format
    fqe_ham = fqe.restricted_hamiltonian.RestrictedHamiltonian(
        (h1e, np.einsum("ijlk", -0.5 * of_eris)), e_0=ecore
    )
    return wfn.expectationValue(fqe_ham).real


def build_ipie_wavefunction_from_pyscf(
    fcivec: np.ndarray, mc: Union[pyscf.mcscf.CASCI, pyscf.mcscf.CASSCF], tol: float = 1e-12
) -> ParticleHole:
    """Build ipie wavefunction in the full space (i.e. with "melting cores")"""
    coeff, occa, occb = zip(
        *pyscf.fci.addons.large_ci(fcivec, mc.ncas, mc.nelecas, tol=tol, return_strs=False)
    )
    ix = np.argsort(np.abs(coeff))[::-1]
    nelec = mc._scf.mol.nelec
    nmo = mc._scf.mo_coeff.shape[-1]
    return ParticleHole((np.array(coeff)[ix], np.array(occa)[ix], np.array(occb)[ix]), nelec, nmo)


def strip_melting_cores(
    occa: np.ndarray, occb: np.ndarray, n_melting: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Strip any melting cores from ipie wavefunction."""
    occa_new = []
    occb_new = []
    for oa, ob in zip(occa, occb):
        # ipie typically builds the cas wavefunction in the full space by inserting "melting" cores
        # To map back to the active space you need to strip these and then shift
        # the orbital indices back by the number of melting cores (n_melting)
        occa_new.append(np.array([o - n_melting for o in oa[n_melting:]]))
        occb_new.append(np.array([o - n_melting for o in ob[n_melting:]]))

    return np.array(occa_new), np.array(occb_new)


def build_ipie_sys_ham_from_pyscf(
    mc: Union[pyscf.mcscf.CASCI, pyscf.mcscf.CASSCF], chol_cut: float = 1e-6
) -> Tuple[GenericSys, GenericHam]:
    """Build ipie system and hamiltonian from MCSCF object."""
    ham = generate_hamiltonian(
        mc._scf.mol,
        mc.mo_coeff,
        mc._scf.get_hcore(),
        mc.mo_coeff,
        chol_cut=chol_cut,
        num_frozen_core=0,
        verbose=False,
    )
    nelec = (mol.nelec[0], mol.nelec[1])
    return GenericSys(nelec), ham


if __name__ == "__main__":
    mol = gto.Mole(atom=[("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.45))], spin=0, basis="sto-3g")
    mol.build()
    mf = scf.RHF(mol)
    mf.kernel()

    nalpha, nbeta = mol.nelec
    nmo = mf.mo_coeff.shape[1]

    ncas, nelecas = (6, 6)
    mc = mcscf.CASSCF(mf, nelecas, ncas)

    e_tot, e_cas, fcivec, mo, mo_energy = mc.kernel()
    print(f"DIM(H) = {fcivec.ravel().shape}")
    # Get the active space ERIs in the CASSCF MO basis
    h1e, e0 = mc.get_h1eff(mc.mo_coeff)
    eris = ao2mo.restore("1", mc.get_h2eff(mc.mo_coeff), ncas).reshape((ncas,) * 4)
    # you can check how truncating the wavefunction affects the energy
    wfn = build_ipie_wavefunction_from_pyscf(fcivec, mc, tol=0.0)
    print(f"Length of truncated CI expansion: {wfn.num_dets}")
    # you check how truncating the Cholesky dimension affects the energy
    sys, ham = build_ipie_sys_ham_from_pyscf(mc, chol_cut=1e-12)

    ipie_energy = wfn.calculate_energy(sys, ham)[0]
    msg = f"{ipie_energy.real:.10f}"
    print(f"ipie energy: {msg}")
    assert np.isclose(e_tot, ipie_energy, atol=1e-8), f"{e_tot} != {msg}"

    # Convert to FQE and check the energy
    occa_fqe, occb_fqe = strip_melting_cores(wfn.occa, wfn.occb, wfn.nmelting)
    fqe_wfn = get_fqe_wfn_from_occ_coeff(
        wfn.coeffs, occa_fqe, occb_fqe, nelecas, ncas, ms=0, threshold=0.0
    )
    fqe_energy = get_fqe_variational_energy(e0, h1e, eris, fqe_wfn)
    msg = f"{fqe_energy.real:.10f}"
    print(f"FQE energy: {msg}")
    assert np.isclose(e_tot, fqe_energy, atol=1e-8), f"{e_tot} != {msg}"

    # round trip back to ipie
    coeff, occa, occb = get_occa_occb_coeff_from_fqe_wfn(fqe_wfn, threshold=0.0)
    wfn_round_trip = ParticleHole((coeff, occa, occb), mc._scf.mol.nelec, mc.mo_coeff.shape[-1])
    ipie_energy = wfn_round_trip.calculate_energy(sys, ham)[0]
    msg = f"{ipie_energy.real:.10f}"
    print(f"ipie energy from round trip: {msg}")
    assert np.isclose(e_tot, ipie_energy, atol=1e-8), f"{e_tot} != {msg}"
