
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
# Authors: Fionn Malone <fionn.malone@gmail.com>
#          Joonho Lee
#

import time

import numpy
import scipy.linalg

from ipie.estimators.generic import (half_rotated_cholesky_hcore,
                                     half_rotated_cholesky_jk)
from ipie.estimators.local_energy import (variational_energy,
                                          variational_energy_ortho_det)
from ipie.legacy.estimators.ci import (get_hmatel, get_one_body_matel,
                                       get_perm, map_orb)
from ipie.legacy.estimators.greens_function import (gab, gab_mod, gab_mod_ovlp,
                                                    gab_spin)
from ipie.legacy.estimators.local_energy import local_energy
from ipie.utils.io import get_input_value, write_qmcpack_wfn
from ipie.utils.mpi import get_shared_array
from ipie.utils.backend import cast_to_device

try:
    from ipie.lib.wicks import wicks_helper
except ImportError:
    wicks_helper = None
    pass

import numpy


class MultiSlater(object):
    def __init__(
        self,
        system,
        hamiltonian,
        wfn,
        nbasis=None,
        options={},
        init=None,
        verbose=False,
        orbs=None,
    ):
        self.verbose = verbose
        if verbose:
            print("# Parsing input options for trial_wavefunction.MultiSlater.")
        init_time = time.time()
        self.name = "MultiSlater"
        self.mixed_precision = hamiltonian.mixed_precision
        self.chunked = False
        # TODO : Fix for MSD.
        # This is for the overlap trial
        self.ortho_expansion = len(wfn) == 3
        self.wicks = get_input_value(
                options,
                "wicks",
                default=self.ortho_expansion,
                verbose=verbose)
        self.optimized = get_input_value(
            options,
            "optimized",
            default=True,
            alias=["optimize", "optimise", "optimised"],
            verbose=verbose,
        )
        if len(wfn) == 3:
            # CI type expansion.
            self.from_phmsd(system.nup, system.ndown, hamiltonian.nbasis, wfn, orbs)
        else:
            psit = wfn[1]
            self.psi = psit
            imag_norm = numpy.sum(self.psi.imag.ravel() * self.psi.imag.ravel())
            if imag_norm <= 1e-8:
                # print("# making trial wavefunction MO coefficient real")
                self.psi = numpy.array(self.psi.real, dtype=numpy.float64)
            self.coeffs = numpy.array(wfn[0], dtype=numpy.complex128)
            self.ortho_expansion = False
            self.nelec_cas = system.nup + system.ndown
            self.nact = hamiltonian.nbasis

        self.psia = self.psi[:, :, : system.nup]
        self.psib = self.psi[:, :, system.nup :]
        self._nalpha = system.nup
        self._nbeta = system.ndown
        self._nelec = system.nelec
        self._nbasis = hamiltonian.nbasis

        self.use_wicks_helper = get_input_value(
            options, "use_wicks_helper", default=False, verbose=verbose
        )

        self.ndets = get_input_value(
            options, "ndets", default=len(self.coeffs), verbose=verbose
        )
        self.compute_trial_energy = get_input_value(
            options, "compute_trial_energy", default=True, verbose=verbose,
            alias=['calculate_variational_energy']
        )
        if self.verbose:
            if self.ortho_expansion:
                print("# Assuming orthogonal trial wavefunction expansion.")
            else:
                print("# Assuming non-orthogonal trial wavefunction expansion.")
            print("# Trial wavefunction shape: {}".format(self.psi.shape))

        # if self.verbose:
        # print("# Setting ndets: {}".format(self.ndets))

        if self.ndets == 1:
            self.G, self.Ghalf = gab_spin(
                self.psi[0], self.psi[0], system.nup, system.ndown
            )
            self.G = numpy.array(self.G, dtype=numpy.complex128)
            self.Ghalf = [
                numpy.array(self.Ghalf[0], dtype=numpy.complex128),
                numpy.array(self.Ghalf[1], dtype=numpy.complex128),
            ]
        else:
            self.G = None
            self.Ghalf = None
        if init is not None:
            if verbose:
                print("# Using initial wavefunction from file.")
            self.init = init
        else:
            if verbose:
                print(
                    "# Setting initial wavefunction as first determinant in"
                    " expansion."
                )
            if len(self.psi.shape) == 3:
                self.init = self.psi[0].copy()
            else:
                self.init = self.psi.copy()

        self.ndets_props = get_input_value(
            options,
            "ndets_for_trial_props",
            default=min(self.ndets, 100),
            alias=["ndets_prop"],
            verbose=verbose,
        )
        self.ndet_chunks = get_input_value(
            options,
            "ndet_chunks",
            default=1,
            alias=["nchunks", "chunks"],
            verbose=verbose,
        )
        assert self.ndet_chunks <= self.ndets, "ndet_chunks > ndets"
        self.nfrozen = (system.nup + system.ndown - self.nelec_cas) // 2
        self.nocc_alpha = system.nup - self.nfrozen
        self.nocc_beta = system.ndown - self.nfrozen
        self.act_orb_alpha = slice(self.nfrozen, self.nfrozen + self.nact)
        self.act_orb_beta = slice(self.nfrozen, self.nfrozen + self.nact)
        self.occ_orb_alpha = slice(self.nfrozen, self.nfrozen + self.nocc_alpha)
        self.occ_orb_beta = slice(self.nfrozen, self.nfrozen + self.nocc_beta)
        if self.wicks:
            if verbose:
                print("# Using generalized Wick's theorem for the PHMSD trial")
                print(
                    "# Setting the first determinant in"
                    " expansion as the reference wfn for Wick's theorem."
                )
                print(f"# Number of frozen orbitals: {self.nfrozen}")
                print(
                    f"# Number of occupied electrons in active space trial: "
                    f"({self.nocc_alpha}, {self.nocc_beta})"
                )
                print(f"# Number of orbitals in active space trial: {self.nact}")
                if self.optimized:
                    # approximate memory for os_buffers and det/cof matrices which are largest
                    # contributors.
                    mem_required = (
                        16
                        * 4
                        * self.ndets
                        * hamiltonian.nchol
                        / (self.ndet_chunks * 1024**3.0)
                    )
                    print(
                        f"# Dominant memory cost **per walker** for optimized "
                        f"energy evaluation using Wick's algorithm: {mem_required} "
                        "GB."
                    )
                    if mem_required > 1.0:
                        print(
                            f"# WARNING: Memory required by energy evaluation "
                            "exceeds 1 GB per walker.\n# Consider increasing ndet_chunks "
                            "option in input file from current value "
                            f"{self.ndet_chunks} to something larger."
                        )
            self.psi0a = self.psi[0, :, : system.nup].copy()
            self.psi0b = self.psi[0, :, system.nup :].copy()
            if verbose:
                print("# Setting additional member variables for Wick's theorem")
            d0a = self.occa[0][self.occ_orb_alpha] - self.nfrozen
            d0b = self.occb[0][self.occ_orb_beta] - self.nfrozen
            if verbose:
                print(f"# Reference alpha determinant: {d0a}")
                print(f"# Reference beta determinant: {d0b}")
            # numba won't accept dictionary in jitted code so use an array so
            # can't do following
            # self.occ_map_a = dict(zip(d0a, list(range(self.nocc_alpha))))
            # self.occ_map_b = dict(zip(d0b, list(range(self.nocc_beta))))

            # Create mapping from reference determinant to occupied orbital
            # index for eg.
            # TODO: Use safer value than zero that fails in debug mode.
            # d0a = [0,1,3,5]
            # occ_map_a = [0,1,0,2,0,3]
            self.occ_map_a = numpy.zeros(max(d0a) + 1, dtype=numpy.int32)
            self.occ_map_b = numpy.zeros(max(d0b) + 1, dtype=numpy.int32)
            self.occ_map_a[d0a] = list(range(self.nocc_alpha))
            self.occ_map_b[d0b] = list(range(self.nocc_beta))
            self.cre_a = [
                []
            ]  # one empty list as a member to account for the reference state
            self.anh_a = [
                []
            ]  # one empty list as a member to account for the reference state
            self.cre_b = [
                []
            ]  # one empty list as a member to account for the reference state
            self.anh_b = [
                []
            ]  # one empty list as a member to account for the reference state
            self.phase_a = numpy.ones(self.ndets)  # 1.0 is for the reference state
            self.phase_b = numpy.ones(self.ndets)  # 1.0 is for the reference state
            nexcit_a = system.nup
            nexcit_b = system.ndown
            # This is an overestimate because we don't know number of active
            # electrons in trial from read in.
            # TODO work this out.
            max_excit = max(nexcit_a, nexcit_b) + 1
            cre_ex_a = [[] for _ in range(max_excit)]
            cre_ex_b = [[] for _ in range(max_excit)]
            anh_ex_a = [[] for _ in range(max_excit)]
            anh_ex_b = [[] for _ in range(max_excit)]
            # Will store mapping from unordered list defined by order in which added to
            # cre_/anh_a/b TO the full determinant index, i.e.,
            # ordered_like_trial[excit_map_a] = buffer_from_cre_ex_a[:]
            excit_map_a = [[] for _ in range(max_excit)]
            excit_map_b = [[] for _ in range(max_excit)]
            num_chunks = self.ndet_chunks
            ndets_chunk = (self.ndets - 1) // num_chunks + 1
            self.ndets_chunk_max = ndets_chunk
            cre_ex_a_chunk = [[[] for _ in range(max_excit)] for i in range(num_chunks)]
            cre_ex_b_chunk = [[[] for _ in range(max_excit)] for i in range(num_chunks)]
            anh_ex_a_chunk = [[[] for _ in range(max_excit)] for i in range(num_chunks)]
            anh_ex_b_chunk = [[[] for _ in range(max_excit)] for i in range(num_chunks)]
            excit_map_a_chunk = [
                [[] for _ in range(max_excit)] for i in range(num_chunks)
            ]
            excit_map_b_chunk = [
                [[] for _ in range(max_excit)] for i in range(num_chunks)
            ]
            for ichunk in range(num_chunks):
                for jdet in range(0, ndets_chunk):
                    j = 1 + ichunk * ndets_chunk + jdet
                    if j == self.ndets:
                        break
                    dja = self.occa[j][self.occ_orb_alpha] - self.nfrozen
                    djb = self.occb[j][self.occ_orb_beta] - self.nfrozen

                    anh_a = list(
                        set(dja) - set(d0a)
                    )  # annihilation to right, creation to left
                    cre_a = list(
                        set(d0a) - set(dja)
                    )  # creation to right, annhilation to left

                    anh_b = list(set(djb) - set(d0b))
                    cre_b = list(set(d0b) - set(djb))

                    cre_a.sort()
                    cre_b.sort()
                    anh_a.sort()
                    anh_b.sort()

                    self.anh_a += [anh_a]
                    self.anh_b += [anh_b]
                    self.cre_a += [cre_a]
                    self.cre_b += [cre_b]
                    anh_ex_a[len(anh_a)].append(anh_a)
                    anh_ex_b[len(anh_b)].append(anh_b)
                    cre_ex_a[len(cre_a)].append(cre_a)
                    cre_ex_b[len(cre_b)].append(cre_b)
                    anh_ex_a_chunk[ichunk][len(anh_a)].append(anh_a)
                    anh_ex_b_chunk[ichunk][len(anh_b)].append(anh_b)
                    cre_ex_a_chunk[ichunk][len(cre_a)].append(cre_a)
                    cre_ex_b_chunk[ichunk][len(cre_b)].append(cre_b)
                    excit_map_a[len(anh_a)].append(j)
                    excit_map_b[len(anh_b)].append(j)
                    excit_map_a_chunk[ichunk][len(anh_a)].append(j)
                    excit_map_b_chunk[ichunk][len(anh_b)].append(j)

                    perm_a = get_perm(anh_a, cre_a, d0a, dja)
                    perm_b = get_perm(anh_b, cre_b, d0b, djb)

                    if perm_a:
                        self.phase_a[j] = -1
                    else:
                        self.phase_a[j] = +1

                    if perm_b:
                        self.phase_b[j] = -1
                    else:
                        self.phase_b[j] = +1
                if j == self.ndets:
                    break

            self.ndets_per_chunk = [
                sum(len(ex) for ex in cre_ex_a_chunk[ichunk])
                for ichunk in range(num_chunks)
            ]
            assert sum(self.ndets_per_chunk) == self.ndets - 1
            self.ndet_a = [len(ex) for ex in cre_ex_a]
            self.ndet_b = [len(ex) for ex in cre_ex_b]
            self.max_excite_a = max(
                -1 if nd == 0 else i for i, nd in enumerate(self.ndet_a)
            )
            self.max_excite_b = max(
                -1 if nd == 0 else i for i, nd in enumerate(self.ndet_b)
            )
            self.max_excite = max(self.max_excite_a, self.max_excite_b)
            self.cre_ex_a = [numpy.array(ex, dtype=numpy.int32) for ex in cre_ex_a]
            self.cre_ex_b = [numpy.array(ex, dtype=numpy.int32) for ex in cre_ex_b]
            self.anh_ex_a = [numpy.array(ex, dtype=numpy.int32) for ex in anh_ex_a]
            self.anh_ex_b = [numpy.array(ex, dtype=numpy.int32) for ex in anh_ex_b]
            self.cre_ex_a_chunk = [
                [numpy.array(ex, dtype=numpy.int32) for ex in cre_ex_a_chunk[ichunk]]
                for ichunk in range(num_chunks)
            ]
            self.cre_ex_b_chunk = [
                [numpy.array(ex, dtype=numpy.int32) for ex in cre_ex_b_chunk[ichunk]]
                for ichunk in range(num_chunks)
            ]
            self.anh_ex_a_chunk = [
                [numpy.array(ex, dtype=numpy.int32) for ex in anh_ex_a_chunk[ichunk]]
                for ichunk in range(num_chunks)
            ]
            self.anh_ex_b_chunk = [
                [numpy.array(ex, dtype=numpy.int32) for ex in anh_ex_b_chunk[ichunk]]
                for ichunk in range(num_chunks)
            ]
            self.excit_map_a = [
                numpy.array(ex, dtype=numpy.int32) for ex in excit_map_a
            ]
            self.excit_map_b = [
                numpy.array(ex, dtype=numpy.int32) for ex in excit_map_b
            ]
            # Will store array remapping from chunk of data created from
            # cre/anh_chunk to original determinant order sliced appropriately
            # to map chunk index.
            # i.e. buffer_from_cre_a[excit_map_a_chunk[1]] *
            # trial.coeffs[slice[1]] will yield something sensible.
            # Note this is the **inverse** mapping from the non chunked case
            self.excit_map_a_chunk = [
                numpy.argsort(numpy.concatenate(excit_map_a_chunk[ichunk]))
                for ichunk in range(num_chunks)
            ]
            self.excit_map_b_chunk = [
                numpy.argsort(numpy.concatenate(excit_map_b_chunk[ichunk]))
                for ichunk in range(num_chunks)
            ]

            self.slices_alpha, self.slices_beta = self.build_slices()
            (
                self.slices_alpha_chunk,
                self.slices_beta_chunk,
            ) = self.build_slices_chunked()

            if verbose:
                print(f"# Number of alpha determinants at each level: {self.ndet_a}")
                print(f"# Number of beta determinants at each level: {self.ndet_b}")

        self.compute_opdm = options.get("compute_opdm", True)

        if self.ortho_expansion and self.compute_opdm:  # this is for phmsd
            if verbose:
                print("# Computing 1-RDM of the trial wfn for mean-field shift.")
                print(f"# Using first {self.ndets_props} determinants for evaluation.")
            start = time.time()
            if self.use_wicks_helper:
                assert wicks_helper is not None
                dets = wicks_helper.encode_dets(self.occa, self.occb)
                phases = wicks_helper.convert_phase(self.occa, self.occb)
                _keep = self.ndets_props
                self.G = wicks_helper.compute_opdm(
                    phases[:_keep] * self.coeffs[:_keep].copy(),
                    dets[:_keep],
                    hamiltonian.nbasis,
                    system.ne,
                )
            else:
                self.G = self.compute_1rdm(hamiltonian.nbasis)
            end = time.time()
            if verbose:
                print("# Time to compute 1-RDM: {} s".format(end - start))

        self.error = False
        self.initialisation_time = time.time() - init_time
        self.half_rotated = False
        self.e1b = None
        self.e2b = None
        self._rchol = None
        self._rH1a = None  # rotated H1
        self._rH1b = None  # rotated H1
        self._rchola = None
        self._rcholb = None
        self._eri = None
        self._mem_required = 0.0

        write_wfn = options.get("write_wavefunction", False)
        output_file = options.get("output_file", "wfn.h5")

        if write_wfn:
            self.write_wavefunction(filename=output_file)
        if verbose:
            print("# Finished setting up trial_wavefunction.MultiSlater.")

    def build_slices(self):
        slices_beta = []
        slices_alpha = []
        start_alpha = 1
        start_beta = 1
        for i in range(0, self.max_excite + 1):
            nd = len(self.cre_ex_a[i])
            slices_alpha.append(slice(start_alpha, start_alpha + nd))
            start_alpha += nd
            nd = len(self.cre_ex_b[i])
            slices_beta.append(slice(start_beta, start_beta + nd))
            start_beta += nd

        return slices_alpha, slices_beta

    def build_slices_chunked(self):
        slices_beta_chunk = []
        slices_alpha_chunk = []
        for ichunk in range(self.ndet_chunks):
            slices_beta = []
            slices_alpha = []
            start_alpha = 0
            start_beta = 0
            for i in range(0, self.max_excite + 1):
                nd = len(self.cre_ex_a_chunk[ichunk][i])
                slices_alpha.append(slice(start_alpha, start_alpha + nd))
                start_alpha += nd
                nd = len(self.cre_ex_b_chunk[ichunk][i])
                slices_beta.append(slice(start_beta, start_beta + nd))
                start_beta += nd
            slices_alpha_chunk.append(slices_alpha)
            slices_beta_chunk.append(slices_beta)

        return slices_alpha_chunk, slices_beta_chunk

    def calculate_energy(self, system, hamiltonian):
        if self.verbose:
            print("# Computing trial wavefunction energy.")
        start = time.time()
        # Cannot use usual energy evaluation routines if trial is orthogonal.
        if self.ortho_expansion:
            self.energy, self.e1b, self.e2b = variational_energy_ortho_det(
                system, hamiltonian, self.spin_occs, self.coeffs
            )
        else:
            # (self.energy, self.e1b, self.e2b) = local_energy_generic_cholesky_opt(system, hamiltonian.ecore, Ghalfa=self.Ghalf[0], Ghalfb=self.Ghalf[1], trial=self)
            self.e1b = (
                numpy.sum(self.Ghalf[0] * self._rH1a)
                + numpy.sum(self.Ghalf[1] * self._rH1b)
                + hamiltonian.ecore
            )
            self.ej, self.ek = half_rotated_cholesky_jk(
                system, self.Ghalf[0], self.Ghalf[1], trial=self
            )
            self.e2b = self.ej + self.ek
            self.energy = self.e1b + self.e2b

            # this is for the correlation energy trick
            # self.e1b_corr = numpy.sum(self.Ghalf[0]*self._rH1a_corr) + numpy.sum(self.Ghalf[1]*self._rH1b_corr) + hamiltonian.ecore
            # self.e2b_corr = self.ej+ self.ek

        if self.verbose:
            print(
                "# (E, E1B, E2B): (%13.8e, %13.8e, %13.8e)"
                % (self.energy.real, self.e1b.real, self.e2b.real)
            )
            print("# Time to evaluate local energy: {} s".format(time.time() - start))

    def from_phmsd(self, nup, ndown, nbasis, wfn, orbs):
        ndets = len(wfn[0])
        ne = nup + ndown
        if self.wicks:
            self.psi = numpy.zeros((1, nbasis, ne), dtype=numpy.float64)
        else:
            self.psi = numpy.zeros((ndets, nbasis, ne), dtype=numpy.float64)
        if self.verbose:
            print("# Creating trial wavefunction from CI-like expansion.")
        if orbs is None:
            if self.verbose:
                print("# Assuming RHF reference.")
            I = numpy.eye(nbasis, dtype=numpy.float64)
        nocca_in_wfn = len(wfn[1][0])
        noccb_in_wfn = len(wfn[2][0])
        max_orbital = max(numpy.max(wfn[1]), numpy.max(wfn[2])) + 1
        if self.optimized:
            self.nelec_cas = nocca_in_wfn + noccb_in_wfn
            self.nact = max_orbital
        else:
            self.nelec_cas = ne
            self.nact = nbasis
        if self.verbose:
            print(f"# Trial wavefunction correlates {self.nelec_cas} electrons "
                  f"in {self.nact} orbitals.")
        if nup != nocca_in_wfn and ndown != noccb_in_wfn:
            occa0 = wfn[1]
            occb0 = wfn[2]
            assert nocca_in_wfn < nup and noccb_in_wfn < ndown
            nmelting_a = nup - nocca_in_wfn
            nmelting_b = ndown - noccb_in_wfn
            num_melting = nmelting_a
            if self.verbose:
                print("# Trial wavefunction contains different number of "
                      " electrons than specified in input file.")
                print(f"# Inserting {num_melting} melting cores.")
            assert nmelting_a == nmelting_b
            core = [i for i in range(num_melting)]
            occa = [numpy.array(core + [o + num_melting for o in oa]) for oa in occa0]
            occb = [numpy.array(core + [o + num_melting for o in ob]) for ob in occb0]
        else:
            occa = wfn[1]
            occb = wfn[2]
        # Store alpha electrons first followed by beta electrons.
        nb = nbasis
        dets = [list(a) + [i + nb for i in c] for (a, c) in zip(occa, occb)]
        self.spin_occs = [numpy.sort(d) for d in dets]
        self.occa = numpy.array(occa, dtype=numpy.int32)
        self.occb = numpy.array(occb, dtype=numpy.int32)
        self.coeffs = numpy.array(wfn[0], dtype=numpy.complex128)
        if self.wicks:
            self.psi[0, :, :nup] = I[:, occa[0]]
            self.psi[0, :, nup:] = I[:, occb[0]]
        else:
            for idet, (oa, ob) in enumerate(zip(occa, occb)):
                self.psi[idet, :, :nup] = I[:, oa]
                self.psi[idet, :, nup:] = I[:, ob]

    def recompute_ci_coeffs(self, nup, ndown, ham):
        H = numpy.zeros((self.ndets, self.ndets), dtype=numpy.complex128)
        S = numpy.zeros((self.ndets, self.ndets), dtype=numpy.complex128)
        m = ham.nbasis
        na = nup
        nb = ndown
        if self.ortho_expansion:
            for i in range(self.ndets):
                for j in range(i, self.ndets):
                    di = self.spin_occs[i]
                    dj = self.spin_occs[j]
                    H[i, j] = get_hmatel(ham, nup + ndown, di, dj)[0]
            e, ev = scipy.linalg.eigh(H, lower=False)
        else:
            na = nup
            for i, di in enumerate(self.psi):
                for j, dj in enumerate(self.psi):
                    if j >= i:
                        ga, gha, ioa = gab_mod_ovlp(di[:, :na], dj[:, :na])
                        gb, ghb, iob = gab_mod_ovlp(di[:, na:], dj[:, na:])
                        G = numpy.array([ga, gb])
                        Ghalf = numpy.array([gha, ghb])
                        ovlp = 1.0 / (scipy.linalg.det(ioa) * scipy.linalg.det(iob))
                        if abs(ovlp) > 1e-12:
                            if self._rchol is not None:
                                rchol = self.rchol(i)
                            else:
                                rchol = None
                            H[i, j] = (
                                ovlp * local_energy(ham, G, Ghalf=Ghalf, rchol=rchol)[0]
                            )
                            S[i, j] = ovlp
                            H[j, i] = numpy.conjugate(H[i, j])
                            S[j, i] = numpy.conjugate(S[i, j])
            e, ev = scipy.linalg.eigh(H, S, lower=False)
        # if self.verbose:
        # print("Old and New CI coefficients: ")
        # for co,cn in zip(self.coeffs,ev[:,0]):
        # print("{} {}".format(co, cn))
        return numpy.array(ev[:, 0], dtype=numpy.complex128)

    def compute_1rdm(self, nbasis):
        assert self.ortho_expansion == True
        denom = numpy.sum(self.coeffs.conj() * self.coeffs)
        Pa = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)
        Pb = numpy.zeros((nbasis, nbasis), dtype=numpy.complex128)
        P = [Pa, Pb]
        for idet in range(self.ndets_props):
            di = self.spin_occs[idet]
            # zero excitation case
            for iorb in range(len(di)):
                ii, spin_ii = map_orb(di[iorb], nbasis)
                P[spin_ii][ii, ii] += self.coeffs[idet].conj() * self.coeffs[idet]
            for jdet in range(idet + 1, self.ndets_props):
                dj = self.spin_occs[jdet]
                from_orb = list(set(dj) - set(di))
                to_orb = list(set(di) - set(dj))
                nex = len(from_orb)
                if nex > 1:
                    continue
                elif nex == 1:
                    perm = get_perm(from_orb, to_orb, di, dj)
                    if perm:
                        phase = -1
                    else:
                        phase = 1
                    ii, si = map_orb(from_orb[0], nbasis)
                    aa, sa = map_orb(to_orb[0], nbasis)
                    if si == sa:
                        P[si][aa, ii] += (
                            self.coeffs[jdet].conj() * self.coeffs[idet] * phase
                        )
                        P[si][ii, aa] += (
                            self.coeffs[jdet] * self.coeffs[idet].conj() * phase
                        )
        P[0] /= denom
        P[1] /= denom

        return P

    def chunk(self, handler, verbose=False):
        self.chunked = True  # Boolean to indicate that chunked cholesky is available

        if handler.scomm.rank == 0:  # Creating copy for every rank == 0
            self._rchola = self._rchola.copy()
            self._rcholb = self._rcholb.copy()

        self._rchola_chunk = handler.scatter_group(self._rchola)  # distribute over chol
        self._rcholb_chunk = handler.scatter_group(self._rcholb)  # distribute over chol

        tot_size = handler.allreduce_group(self._rchola_chunk.size)
        assert self._rchola.size == tot_size
        tot_size = handler.allreduce_group(self._rcholb_chunk.size)
        assert self._rcholb.size == tot_size

    # This function casts relevant member variables into cupy arrays
    def cast_to_cupy(self, verbose=False):
        cast_to_device(self, verbose)

    def contract_one_body(self, ints):
        numer = 0.0
        denom = 0.0
        na = self._nalpha
        for i in range(self.ndets):
            for j in range(self.ndets):
                cfac = self.coeffs[i].conj() * self.coeffs[j]
                if self.ortho_expansion:
                    di = self.spin_occs[i]
                    dj = self.spin_occs[j]
                    tij = get_one_body_matel(ints, di, dj)
                    numer += cfac * tij
                    if i == j:
                        denom += self.coeffs[i].conj() * self.coeffs[i]
                else:
                    di = self.psi[i]
                    dj = self.psi[j]
                    ga, gha, ioa = gab_mod_ovlp(di[:, :na], dj[:, :na])
                    gb, ghb, iob = gab_mod_ovlp(di[:, na:], dj[:, na:])
                    ovlp = 1.0 / (scipy.linalg.det(ioa) * scipy.linalg.det(iob))
                    tij = numpy.dot(ints.ravel(), ga.ravel() + gb.ravel())
                    numer += cfac * ovlp * tij
                    denom += cfac * ovlp
        return numer / denom

    def write_wavefunction(self, filename="wfn.h5", init=None, occs=False):
        if occs:
            wfn = (self.coeffs, self.occa, self.occb)
        else:
            wfn = (self.coeffs, self.psi)
        write_qmcpack_wfn(filename, wfn, "uhf", self._nelec, self._nbasis, init=init)

    def half_rotate(self, system, hamiltonian, comm=None):
        if self.verbose:
            print("# Constructing half rotated Cholesky vectors.")
        M = hamiltonian.nbasis
        nchol = hamiltonian.nchol
        self.half_rotated = True
        # Half rotated cholesky vectors (by trial wavefunction or a reference wfn in the case of PHMSD).
        na = system.nup
        nb = system.ndown
        psi = self.psi[0]
        if self.verbose:
            print("# Shape of alpha half-rotated Cholesky: {}".format((nchol, na * M)))
            print("# Shape of beta half-rotated Cholesky: {}".format((nchol, nb * M)))

        # assert self.ortho_expansion
        if self.verbose:
            print("# Only performing half-rotation of the reference determinant")
        hr_ndet = 1

        if isinstance(hamiltonian.chol_vecs, numpy.ndarray):
            chol = hamiltonian.chol_vecs.reshape((M, M, nchol))
        else:
            chol = hamiltonian.chol_vecs.toarray().reshape((nchol, M, M))

        if hamiltonian.exact_eri:
            shape = (hr_ndet, (M**2 * (na**2 + nb**2) + M**2 * (na * nb)))
            self._eri = get_shared_array(comm, shape, numpy.complex128)
            self._mem_required = self._eri.nbytes / (1024.0**3.0)

            vipjq_aa = numpy.einsum(
                "Xmp,Xrq,mi,rj->ipjq",
                chol,
                chol,
                psi[:, :na].conj(),
                psi[:, :na].conj(),
                optimize=True,
            )
            vipjq_bb = numpy.einsum(
                "Xmp,Xrq,mi,rj->ipjq",
                chol,
                chol,
                psi[:, na:].conj(),
                psi[:, na:].conj(),
                optimize=True,
            )
            vipjq_ab = numpy.einsum(
                "Xmp,Xrq,mi,rj->ipjq",
                chol,
                chol,
                psi[:, :na].conj(),
                psi[:, na:].conj(),
                optimize=True,
            )
            self._eri[i, : M**2 * na**2] = vipjq_aa.ravel()
            self._eri[
                i, M**2 * na**2 : M**2 * na**2 + M**2 * nb**2
            ] = vipjq_bb.ravel()
            self._eri[i, M**2 * na**2 + M**2 * nb**2 :] = vipjq_ab.ravel()

            if self.verbose:
                print(
                    "# Memory required by exact ERIs: "
                    " {:.4f} GB.".format(self._mem_required)
                )
            if comm is not None:
                comm.barrier()

        shape_a = (nchol, hr_ndet * (M * (na)))
        shape_b = (nchol, hr_ndet * (M * (nb)))
        self._rchola = get_shared_array(comm, shape_a, self.psi.dtype)
        self._rcholb = get_shared_array(comm, shape_b, self.psi.dtype)
        build_act_chol = self.wicks and self.optimized
        if build_act_chol:
            shape_a_act = (nchol, (M * (self.nact)))
            shape_b_act = (nchol, (M * (self.nact)))
            nact = self.nact
            nfrz = self.nfrozen
            self._rchola_act = get_shared_array(comm, shape_a_act, self.psi.dtype)
            self._rcholb_act = get_shared_array(comm, shape_b_act, self.psi.dtype)
            if build_act_chol:
                # working in MO basis so can just grab correct slice
                chol_act = chol[nfrz : nfrz + nact].reshape((M * self.nact, -1))
                self._rchola_act[:] = chol_act.T.copy()
                chol_act = chol[nfrz : nfrz + nact].reshape((M * self.nact, -1))
                self._rcholb_act[:] = chol_act.T.copy()

        self._rH1a = get_shared_array(comm, (na, M), self.psi.dtype)
        self._rH1b = get_shared_array(comm, (nb, M), self.psi.dtype)

        self._rH1a = psi[:, :na].conj().T.dot(hamiltonian.H1[0])
        self._rH1b = psi[:, na:].conj().T.dot(hamiltonian.H1[1])

        self._rH1a = self._rH1a.reshape(na, M)
        self._rH1b = self._rH1b.reshape(nb, M)

        start_time = time.time()
        if self.verbose:
            print("# Half-Rotating Cholesky for determinant.")
        # start = i*M*(na+nb)
        start_a = 0  # determinant loops
        start_b = 0
        compute = True
        # Distribute amongst MPI tasks on this node.
        if comm is not None:
            nwork_per_thread = hamiltonian.nchol // comm.size
            if nwork_per_thread == 0:
                start_n = 0
                end_n = nchol
                if comm.rank != 0:
                    # Just run on root processor if problem too small.
                    compute = False
            else:
                start_n = comm.rank * nwork_per_thread  # Cholesky work split
                end_n = (comm.rank + 1) * nwork_per_thread
                if comm.rank == comm.size - 1:
                    end_n = nchol
        else:
            start_n = 0
            end_n = hamiltonian.nchol

        nchol_loc = end_n - start_n
        # if comm.rank == 0:
        # print(start_n, end_n, nchol_loc)
        # print(numpy.may_share_memory(chol, chol[:,start_n:end_n]))
        if compute:
            # Investigate whether these einsums are fast in the future
            rup = numpy.einsum(
                "mi,mnx->xin",
                psi[:, :na].conj(),
                chol[:, :, start_n:end_n],
                optimize=True,
            )
            rup = rup.reshape((nchol_loc, na * M))
            rdn = numpy.einsum(
                "mi,mnx->xin",
                psi[:, na:].conj(),
                chol[:, :, start_n:end_n],
                optimize=True,
            )
            rdn = rdn.reshape((nchol_loc, nb * M))
            self._rchola[start_n:end_n, start_a : start_a + M * na] = rup[:]
            self._rcholb[start_n:end_n, start_b : start_b + M * nb] = rdn[:]

        self._mem_required = (self._rchola.nbytes + self._rcholb.nbytes) / (
            1024.0**3.0
        )
        self._mem_required += (self._rH1a.nbytes + self._rH1b.nbytes) / (1024.0**3.0)
        if self.verbose:
            print(
                "# Memory required by half-rotated integrals: "
                " {:.4f} GB.".format(self._mem_required)
            )
            print(
                "# Time to half-rotated integrals: {} s.".format(
                    time.time() - start_time
                )
            )
        if comm is not None:
            comm.barrier()

        # storing intermediates for correlation energy
        self._rchola = self._rchola.reshape(hamiltonian.nchol, na, M)
        self._rcholb = self._rcholb.reshape(hamiltonian.nchol, nb, M)

        if hamiltonian.density_diff:
            start_time = time.time()
            Xa = numpy.einsum("mi,xim->x", psi[:, :na], self._rchola, optimize=True)
            Xb = numpy.einsum("mi,xim->x", psi[:, na:], self._rcholb, optimize=True)
            X = Xa + Xb
            J0a = numpy.einsum("x,xim->im", X, self._rchola, optimize=True)  # occ x M
            J0b = numpy.einsum("x,xim->im", X, self._rcholb, optimize=True)  # occ x M

            Ka = numpy.einsum("xim,xin->mn", self._rchola, self._rchola)
            Kb = numpy.einsum("xim,xin->mn", self._rcholb, self._rcholb)
            K0a = self.psi[0, :, :na].T.conj().dot(Ka)  # occ x M
            K0b = self.psi[0, :, na:].T.conj().dot(Kb)  # occ x M

            self._rH1a_corr = get_shared_array(comm, (hr_ndet, na, M), self.psi.dtype)
            self._rH1b_corr = get_shared_array(comm, (hr_ndet, nb, M), self.psi.dtype)

            self._rH1a_corr = self._rH1a + J0a - K0a
            self._rH1b_corr = self._rH1b + J0b - K0b
            self._rFa_corr = J0a - K0a
            self._rFb_corr = J0b - K0b
            if self.verbose:
                print(
                    "# Time to form intermediates for the density difference trick: {} s.".format(
                        time.time() - start_time
                    )
                )

        self._rchola = self._rchola.reshape(hamiltonian.nchol, na * M)
        self._rcholb = self._rcholb.reshape(hamiltonian.nchol, nb * M)

        if self.mixed_precision:
            self._vbias0 = self._rchola.dot(psi[:, :na].T.ravel()) + self._rchola.dot(
                psi[:, na:].T.ravel()
            )
            self._rchola = self._rchola.astype(numpy.float32)
            self._rcholb = self._rcholb.astype(numpy.float32)

    def rot_chol(self, idet=0, spin=None):
        """Helper function"""
        if spin is None:
            print("# rot_chol no longer supported without spin ")
            exit()
            stride = self._nbasis * (self._nalpha + self._nbeta)
            return self._rchol[idet * stride : (idet + 1) * stride]
        else:
            # stride = self._nbasis * (self._nalpha + self._nbeta)
            if spin == 0:
                alpha = self._nbasis * self._nalpha
                stride = self._nbasis * self._nalpha
                return self._rchola[:, idet * stride : idet * stride + alpha]
            else:
                beta = self._nbasis * self._nbeta
                stride = self._nbasis * self._nbeta
                # return self._rchol[idet*stride+alpha:idet*stride+alpha+beta]
                return self._rcholb[:, idet * stride : idet * stride + beta]
