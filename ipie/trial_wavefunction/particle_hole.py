import numpy as np
from typing import Tuple
import time

from ipie.estimators.local_energy import variational_energy_ortho_det
from ipie.legacy.estimators.ci import get_perm
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.trial_wavefunction.half_rotate import half_rotate_generic

# FDM Clean this up!
try:
    from ipie.lib.wicks import wicks_helper

    _use_wicks_helper = True
except ImportError:
    _use_wicks_helper = False
    pass


class ParticleHoleWicks(TrialWavefunctionBase):
    def __init__(
        self,
        wfn,
        nelec,
        nbasis,
        num_dets_for_props=100,
        num_dets_for_trial=-1,
        num_det_chunks=1,
        verbose=False,
    ) -> None:
        super().__init__(
            wfn,
            nelec,
            nbasis,
            verbose=verbose,
        )
        self.setup_basic_wavefunction(wfn, num_dets=num_dets_for_trial)
        self._num_dets_for_props = num_dets_for_props
        self._num_dets = len(self.coeffs)
        self._num_dets_for_props = num_dets_for_props
        self._num_dets_for_trial = num_dets_for_trial
        self._num_det_chunks = num_det_chunks

    def setup_basic_wavefunction(self, wfn, num_dets=None):
        """Unpack wavefunction and insert melting core orbitals."""
        nalpha, nbeta = self.nelec
        ne = sum(self.nelec)
        assert len(wfn) == 3
        self._max_num_dets = len(wfn[0])
        if num_dets == -1:
            num_dets = len(wfn[0])
        nocca_in_wfn = len(wfn[1][0])
        noccb_in_wfn = len(wfn[2][0])
        if nalpha != nocca_in_wfn and nbeta != noccb_in_wfn:
            occa0 = wfn[1][:num_dets]
            occb0 = wfn[2][:num_dets]
            assert nocca_in_wfn < nalpha and noccb_in_wfn < nbeta
            nmelting_a = nalpha - nocca_in_wfn
            nmelting_b = nbeta - noccb_in_wfn
            num_melting = nmelting_a
            if self.verbose:
                print(
                    "# Trial wavefunction contains different number of "
                    " electrons than specified in input file."
                )
                print(f"# Inserting {num_melting} melting cores.")
            assert nmelting_a == nmelting_b
            core = [i for i in range(num_melting)]
            occa = [np.array(core + [o + num_melting for o in oa]) for oa in occa0]
            occb = [np.array(core + [o + num_melting for o in ob]) for ob in occb0]
        else:
            occa = wfn[1][:num_dets]
            occb = wfn[2][:num_dets]
        # Store alpha electrons first followed by beta electrons.
        # FDM Remove this with wicks helper proper integration
        dets = [list(a) + [i + self.nbasis for i in c] for (a, c) in zip(occa, occb)]
        self.spin_occs = [np.sort(d) for d in dets]
        self.occa = np.array(occa, dtype=np.int32)
        self.occb = np.array(occb, dtype=np.int32)
        self.coeffs = np.array(wfn[0][:num_dets], dtype=np.complex128)
        identity = np.eye(self.nbasis, dtype=np.complex128)
        self.nelec_cas = nocca_in_wfn + noccb_in_wfn
        max_orbital = max(np.max(wfn[1]), np.max(wfn[2])) + 1
        self.nact = max_orbital
        self.nfrozen = (nalpha + nbeta - self.nelec_cas) // 2
        self.nact = self.nbasis
        self.nocc_alpha = nalpha - self.nfrozen
        self.nocc_beta = nbeta - self.nfrozen
        self.act_orb_alpha = slice(self.nfrozen, self.nfrozen + self.nact)
        self.act_orb_beta = slice(self.nfrozen, self.nfrozen + self.nact)
        self.occ_orb_alpha = slice(self.nfrozen, self.nfrozen + self.nocc_alpha)
        self.occ_orb_beta = slice(self.nfrozen, self.nfrozen + self.nocc_beta)
        if self.verbose:
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
            # approximate memory for os_buffers and det/cof matrices which are largest
            # contributors.
        self.psi0a = identity[:, self.occa[0]].copy()
        self.psi0b = identity[:, self.occb[0]].copy()

    @property
    def num_det_chunks(self) -> int:
        return self._num_det_chunks

    @num_det_chunks.setter
    def num_det_chunks(self, nchunks: int) -> None:
        self._num_det_chunks = nchunks
        if self._num_det_chunks <= self.num_dets:
            raise RuntimeError(
                f"Requested more determinant chunks than there are determinants"
                "wavefunction. {self._num_det_chunks} vs {self.num_dets}"
            )

    @property
    def num_dets_for_props(self) -> int:
        return self._num_dets_for_props

    @num_dets_for_props.setter
    def num_dets_for_props(self, ndets_props: int) -> None:
        self._num_dets_for_props = ndets_props
        if self._num_dets_for_props > self.num_dets:
            raise RuntimeError(
                "Requested more determinants for property evaluation than"
                "there are in wavefunction"
            )

    def build(
        self,
    ):
        if self.verbose:
            print("# Setting additional member variables for Wick's theorem")
        d0a = self.occa[0][self.occ_orb_alpha] - self.nfrozen
        d0b = self.occb[0][self.occ_orb_beta] - self.nfrozen
        if self.verbose:
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
        self.occ_map_a = np.zeros(max(d0a) + 1, dtype=np.int32)
        self.occ_map_b = np.zeros(max(d0b) + 1, dtype=np.int32)
        self.occ_map_a[d0a] = list(range(self.nocc_alpha))
        self.occ_map_b[d0b] = list(range(self.nocc_beta))
        self.phase_a = np.ones(self.num_dets)  # 1.0 is for the reference state
        self.phase_b = np.ones(self.num_dets)  # 1.0 is for the reference state
        nalpha, nbeta = self.nelec
        nexcit_a = nalpha
        nexcit_b = nbeta
        # This is an overestimate because we don't know number of active
        # electrons in trial from read in.
        # TODO work this out.
        max_excit = max(nexcit_a, nexcit_b) + 1
        num_chunks = self.num_det_chunks
        ndets_chunk = (self.num_dets - 1) // num_chunks + 1
        self.ndets_chunk_max = ndets_chunk
        cre_ex_a_chunk = [[[] for _ in range(max_excit)] for i in range(num_chunks)]
        cre_ex_b_chunk = [[[] for _ in range(max_excit)] for i in range(num_chunks)]
        anh_ex_a_chunk = [[[] for _ in range(max_excit)] for i in range(num_chunks)]
        anh_ex_b_chunk = [[[] for _ in range(max_excit)] for i in range(num_chunks)]
        excit_map_a_chunk = [[[] for _ in range(max_excit)] for i in range(num_chunks)]
        excit_map_b_chunk = [[[] for _ in range(max_excit)] for i in range(num_chunks)]
        for ichunk in range(num_chunks):
            for jdet in range(0, ndets_chunk):
                j = 1 + ichunk * ndets_chunk + jdet
                if j == self.num_dets:
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

                anh_ex_a_chunk[ichunk][len(anh_a)].append(anh_a)
                anh_ex_b_chunk[ichunk][len(anh_b)].append(anh_b)
                cre_ex_a_chunk[ichunk][len(cre_a)].append(cre_a)
                cre_ex_b_chunk[ichunk][len(cre_b)].append(cre_b)
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
            if j == self.num_dets:
                break

        self.ndets_per_chunk = [
            sum(len(ex) for ex in cre_ex_a_chunk[ichunk])
            for ichunk in range(num_chunks)
        ]
        assert sum(self.ndets_per_chunk) == self.num_dets - 1
        self.ndet_a = [
            sum([len(cre_ex_a_chunk[ichunk][iex]) for ichunk in range(num_chunks)])
            for iex in range(max_excit)
        ]
        self.ndet_b = [
            sum([len(cre_ex_b_chunk[ichunk][iex]) for ichunk in range(num_chunks)])
            for iex in range(max_excit)
        ]
        self.max_excite_a = max(
            -1 if nd == 0 else i for i, nd in enumerate(self.ndet_a)
        )
        self.max_excite_b = max(
            -1 if nd == 0 else i for i, nd in enumerate(self.ndet_b)
        )
        self.max_excite = max(self.max_excite_a, self.max_excite_b)
        self.cre_ex_a_chunk = [
            [np.array(ex, dtype=np.int32) for ex in cre_ex_a_chunk[ichunk]]
            for ichunk in range(num_chunks)
        ]
        self.cre_ex_b_chunk = [
            [np.array(ex, dtype=np.int32) for ex in cre_ex_b_chunk[ichunk]]
            for ichunk in range(num_chunks)
        ]
        self.anh_ex_a_chunk = [
            [np.array(ex, dtype=np.int32) for ex in anh_ex_a_chunk[ichunk]]
            for ichunk in range(num_chunks)
        ]
        self.anh_ex_b_chunk = [
            [np.array(ex, dtype=np.int32) for ex in anh_ex_b_chunk[ichunk]]
            for ichunk in range(num_chunks)
        ]
        # Will store array remapping from chunk of data created from
        # cre/anh_chunk to original determinant order sliced appropriately
        # to map chunk index.
        # i.e. buffer_from_cre_a[excit_map_a_chunk[1]] *
        # trial.coeffs[slice[1]] will yield something sensible.
        # Note this is the **inverse** mapping from the non chunked case
        self.excit_map_a_chunk = [
            np.argsort(np.concatenate(excit_map_a_chunk[ichunk]))
            for ichunk in range(num_chunks)
        ]
        self.excit_map_b_chunk = [
            np.argsort(np.concatenate(excit_map_b_chunk[ichunk]))
            for ichunk in range(num_chunks)
        ]

        self.slices_alpha_chunk, self.slices_beta_chunk = self.build_slices_chunked()

        if self.verbose:
            print(f"# Number of alpha determinants at each level: {self.ndet_a}")
            print(f"# Number of beta determinants at each level: {self.ndet_b}")

    def build_slices_chunked(self):
        slices_beta_chunk = []
        slices_alpha_chunk = []
        for ichunk in range(self.num_det_chunks):
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

    def half_rotate(self, system, hamiltonian, comm=None):
        # First get half rotated integrals for reference determinant
        ndets = 1
        orbsa = self.psi0a.reshape((ndets, self.nbasis, self.nalpha))
        orbsb = self.psi0b.reshape((ndets, self.nbasis, self.nbeta))
        rot_1body, rot_chol = half_rotate_generic(
            self,
            system,
            hamiltonian,
            comm,
            orbsa,
            orbsb,
            ndets=ndets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rH1a = rot_1body[0][0]
        self._rH1b = rot_1body[1][0]
        self._rchola = rot_chol[0][0]
        self._rcholb = rot_chol[1][0]
        # In MO basis just need to pick out active orbitals.
        Id = np.eye(self.nbasis, dtype=self.psi0a.dtype)
        act_orbs = (
            Id[:, self.nfrozen : self.nfrozen + self.nact]
            .copy()
            .reshape((ndets, self.nbasis, self.nact))
        )
        _, rot_chol_act = half_rotate_generic(
            self,
            system,
            hamiltonian,
            comm,
            act_orbs,
            act_orbs,
            ndets=ndets,
            verbose=self.verbose,
        )
        # Single determinant functions do not expect determinant index, so just
        # grab zeroth element.
        self._rchola_act = rot_chol_act[0][0]
        # Discared beta since not needed.
        self._rcholb_act = rot_chol_act[0][0]


# No chunking no excitation data structure
class ParticleHoleWicksNonChunked(ParticleHoleWicks):
    def __init__(
        self,
        wfn,
        nelec,
        nbasis,
        num_dets_for_props=100,
        num_dets_for_trial=-1,
        verbose=False,
    ) -> None:
        super().__init__(
            wfn,
            nelec,
            nbasis,
            num_dets_for_props=num_dets_for_props,
            num_dets_for_trial=num_dets_for_trial,
            verbose=verbose,
        )

    def build(
        self,
    ):
        if self.verbose:
            print("# Setting additional member variables for Wick's theorem")
        d0a = self.occa[0][self.occ_orb_alpha] - self.nfrozen
        d0b = self.occb[0][self.occ_orb_beta] - self.nfrozen
        if self.verbose:
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
        self.occ_map_a = np.zeros(max(d0a) + 1, dtype=np.int32)
        self.occ_map_b = np.zeros(max(d0b) + 1, dtype=np.int32)
        self.occ_map_a[d0a] = list(range(self.nocc_alpha))
        self.occ_map_b[d0b] = list(range(self.nocc_beta))
        self.phase_a = np.ones(self.num_dets)  # 1.0 is for the reference state
        self.phase_b = np.ones(self.num_dets)  # 1.0 is for the reference state
        nalpha, nbeta = self.nelec
        nexcit_a = nalpha
        nexcit_b = nbeta
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
        for j in range(1, self.num_dets):
            if j == self.num_dets:
                break
            dja = self.occa[j][self.occ_orb_alpha] - self.nfrozen
            djb = self.occb[j][self.occ_orb_beta] - self.nfrozen

            anh_a = list(set(dja) - set(d0a))  # annihilation to right, creation to left
            cre_a = list(set(d0a) - set(dja))  # creation to right, annhilation to left

            anh_b = list(set(djb) - set(d0b))
            cre_b = list(set(d0b) - set(djb))

            cre_a.sort()
            cre_b.sort()
            anh_a.sort()
            anh_b.sort()

            anh_ex_a[len(anh_a)].append(anh_a)
            anh_ex_b[len(anh_b)].append(anh_b)
            cre_ex_a[len(cre_a)].append(cre_a)
            cre_ex_b[len(cre_b)].append(cre_b)
            excit_map_a[len(anh_a)].append(j)
            excit_map_b[len(anh_b)].append(j)

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

        self.ndet_a = [len(ex) for ex in cre_ex_a]
        self.ndet_b = [len(ex) for ex in cre_ex_b]
        self.max_excite_a = max(
            -1 if nd == 0 else i for i, nd in enumerate(self.ndet_a)
        )
        self.max_excite_b = max(
            -1 if nd == 0 else i for i, nd in enumerate(self.ndet_b)
        )
        self.max_excite = max(self.max_excite_a, self.max_excite_b)
        self.cre_ex_a = [np.array(ex, dtype=np.int32) for ex in cre_ex_a]
        self.cre_ex_b = [np.array(ex, dtype=np.int32) for ex in cre_ex_b]
        self.anh_ex_a = [np.array(ex, dtype=np.int32) for ex in anh_ex_a]
        self.anh_ex_b = [np.array(ex, dtype=np.int32) for ex in anh_ex_b]
        self.excit_map_a = [np.array(ex, dtype=np.int32) for ex in excit_map_a]
        self.excit_map_b = [np.array(ex, dtype=np.int32) for ex in excit_map_b]
        self.slices_alpha, self.slices_beta = self.build_slices()
        if self.verbose:
            print(f"# Number of alpha determinants at each level: {self.ndet_a}")
            print(f"# Number of beta determinants at each level: {self.ndet_b}")

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


class ParticleHoleWicksSlow(ParticleHoleWicks):
    def __init__(
        self,
        wavefunction: tuple,
        num_elec: Tuple[int, int],
        num_basis: int,
        verbose: bool = False,
        num_dets_for_props: int = 100,
        num_dets_for_trial: int = -1,
    ) -> None:
        super().__init__(wavefunction, num_elec, num_basis, verbose=verbose)

    def build(
        self,
    ):
        if self.verbose:
            print("# Setting additional member variables for Wick's theorem")
        d0a = self.occa[0][self.occ_orb_alpha] - self.nfrozen
        d0b = self.occb[0][self.occ_orb_beta] - self.nfrozen
        if self.verbose:
            print(f"# Reference alpha determinant: {d0a}")
            print(f"# Reference beta determinant: {d0b}")
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
        self.phase_a = np.ones(self.num_dets)  # 1.0 is for the reference state
        self.phase_b = np.ones(self.num_dets)  # 1.0 is for the reference state
        nalpha, nbeta = self.nelec
        nexcit_a = nalpha
        nexcit_b = nbeta
        for j in range(1, self.num_dets):
            dja = self.occa[j][self.occ_orb_alpha] - self.nfrozen
            djb = self.occb[j][self.occ_orb_beta] - self.nfrozen

            anh_a = list(set(dja) - set(d0a))  # annihilation to right, creation to left
            cre_a = list(set(d0a) - set(dja))  # creation to right, annhilation to left

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

    def calculate_energy(self, system, hamiltonian):
        if self.verbose:
            print("# Computing trial wavefunction energy.")
        # Cannot use usual energy evaluation routines if trial is orthogonal.
        self.energy, self.e1b, self.e2b = variational_energy_ortho_det(
            system, hamiltonian, self.spin_occs, self.coeffs
        )

    def compute_1rdm(self, nbasis):
        if self.verbose:
            print("# Computing 1-RDM of the trial wfn for mean-field shift.")
            print(
                f"# Using first {self.num_dets_for_props} determinants for evaluation."
            )
        start = time.time()
        assert wicks_helper is not None
        dets = wicks_helper.encode_dets(self.occa, self.occb)
        phases = wicks_helper.convert_phase(self.occa, self.occb)
        _keep = self.num_dets_for_props
        self.G = wicks_helper.compute_opdm(
            phases[:_keep] * self.coeffs[:_keep].copy(),
            dets[:_keep],
            self.nbasis,
            self.nelec,
        )
        end = time.time()
        if self.verbose:
            print("# Time to compute 1-RDM: {} s".format(end - start))
