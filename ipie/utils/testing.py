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

import itertools
from dataclasses import dataclass
from typing import Tuple, Union

import numpy

from ipie.hamiltonians import Generic as HamGeneric
from ipie.propagation.phaseless_generic import PhaselessBase, PhaselessGeneric
from ipie.qmc.afqmc import AFQMC
from ipie.qmc.options import QMCOpts
from ipie.systems import Generic
from ipie.trial_wavefunction.noci import NOCI
from ipie.trial_wavefunction.particle_hole import (
    ParticleHole,
    ParticleHoleNaive,
    ParticleHoleNonChunked,
    ParticleHoleSlow,
)
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.trial_wavefunction.wavefunction_base import TrialWavefunctionBase
from ipie.utils.io import get_input_value
from ipie.utils.linalg import modified_cholesky
from ipie.utils.mpi import MPIHandler
from ipie.walkers.base_walkers import BaseWalkers
from ipie.walkers.pop_controller import PopController
from ipie.walkers.walkers_dispatch import UHFWalkersTrial


def generate_hamiltonian(nmo, nelec, cplx=False, sym=8, tol=1e-3):
    h1e = numpy.random.random((nmo, nmo))
    if cplx:
        h1e = h1e + 1j * numpy.random.random((nmo, nmo))
    eri = numpy.random.normal(scale=0.01, size=(nmo, nmo, nmo, nmo))
    if cplx:
        eri = eri + 1j * numpy.random.normal(scale=0.01, size=(nmo, nmo, nmo, nmo))
    # Restore symmetry to the integrals.
    if sym >= 4:
        # (ik|jl) = (jl|ik)
        # (ik|jl) = (ki|lj)*
        eri = eri + eri.transpose(2, 3, 0, 1)
        eri = eri + eri.transpose(3, 2, 1, 0).conj()

        numpy.testing.assert_allclose(eri, eri.transpose(1, 0, 3, 2).conj(), atol=1e-10)
        numpy.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-10)
        numpy.testing.assert_allclose(eri, eri.transpose(3, 2, 1, 0).conj(), atol=1e-10)

    if sym == 8:
        eri = eri + eri.transpose(1, 0, 2, 3)

        numpy.testing.assert_allclose(eri, eri.transpose(1, 0, 3, 2), atol=1e-10)
        numpy.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-10)
        numpy.testing.assert_allclose(eri, eri.transpose(3, 2, 1, 0), atol=1e-10)
        numpy.testing.assert_allclose(eri, eri.transpose(3, 2, 1, 0), atol=1e-10)
        numpy.testing.assert_allclose(eri, eri.transpose(1, 0, 2, 3), atol=1e-10)
        numpy.testing.assert_allclose(eri, eri.transpose(2, 3, 1, 0), atol=1e-10)
        numpy.testing.assert_allclose(eri, eri.transpose(3, 2, 0, 1), atol=1e-10)

    # Construct hermitian matrix M_{ik,lj}.
    eri = eri.transpose((0, 1, 3, 2))
    eri = eri.reshape((nmo * nmo, nmo * nmo))
    # Make positive semi-definite.
    eri = numpy.dot(eri, eri.conj().T)
    chol = modified_cholesky(eri, tol=tol, verbose=False, cmax=30)
    chol = chol.reshape((-1, nmo, nmo))
    enuc = numpy.random.rand()
    eri = eri.reshape((nmo, nmo, nmo, nmo))
    eri = eri.transpose((0, 1, 3, 2))  # putting it back to the right order for 4-index
    return h1e, chol, enuc, eri


def get_random_nomsd(nup, ndown, nbasis, ndet=10, cplx=True, init=False):
    a = numpy.random.rand(ndet * nbasis * (nup + ndown))
    b = numpy.random.rand(ndet * nbasis * (nup + ndown))
    if cplx:
        wfn = (a + 1j * b).reshape((ndet, nbasis, nup + ndown))
        coeffs = numpy.random.rand(ndet) + 1j * numpy.random.rand(ndet)
    else:
        wfn = a.reshape((ndet, nbasis, nup + ndown))
        coeffs = numpy.random.rand(ndet)
    if init:
        a = numpy.random.rand(nbasis * (nup + ndown))
        b = numpy.random.rand(nbasis * (nup + ndown))
        if cplx:
            init_wfn = (a + 1j * b).reshape((nbasis, nup + ndown))
        else:
            init_wfn = a.reshape((nbasis, nup + ndown))
        return (coeffs, wfn, init_wfn)
    else:
        return (coeffs, wfn)


def truncated_combinations(iterable, r, count):
    # Modified from:
    # https://docs.python.org/3/library/itertools.html#itertools.combinations
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield tuple(pool[i] for i in indices)
    for i in range(count):
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield tuple(pool[i] for i in indices)


def get_random_phmsd(nup, ndown, nbasis, ndet=10, init=False, shuffle=False, cmplx=True):
    orbs = numpy.arange(nbasis)
    oa = [c for c in itertools.combinations(orbs, nup)]
    ob = [c for c in itertools.combinations(orbs, ndown)]
    oa, ob = zip(*itertools.product(oa, ob))

    if shuffle:
        ntot = len(oa)
        det_list = [
            numpy.random.randint(0, ntot - 1) for i in range(ndet)
        ]  # this may pick duplicated list...
        oa = numpy.array(oa)
        ob = numpy.array(ob)
        oa_new = oa[det_list, :]
        ob_new = ob[det_list, :]
        oa = oa_new.copy()
        ob = ob_new.copy()
    else:
        oa = list(oa[:ndet])
        ob = list(ob[:ndet])
    coeffs = numpy.random.rand(ndet) + 1j * numpy.random.rand(ndet)
    wfn = (coeffs, oa, ob)
    if init:
        a = numpy.random.rand(nbasis * (nup + ndown))
        b = numpy.random.rand(nbasis * (nup + ndown))
        if cmplx:
            init_wfn = (a + 1j * b).reshape((nbasis, nup + ndown))
        else:
            init_wfn = a.reshape((nbasis, nup + ndown))
    return wfn, init_wfn


def _gen_det_selection(d0, vir, occ, dist, nel):
    _vir = list(truncated_combinations(vir, nel, dist[nel]))
    _occ = list(truncated_combinations(occ, nel, dist[nel]))
    if len(_vir) == 0 or len(_occ) == 0:
        return None
    ndet = min(dist[nel], len(_vir) * len(_occ))
    occs, virs = zip(*itertools.product(_occ, _vir))
    # choose = numpy.arange(len(occs))
    choose = numpy.random.choice(numpy.arange(len(occs)), ndet, replace=False)
    dets = []
    for ichoose in choose:
        new_det = d0.copy()
        o, v = occs[ichoose], virs[ichoose]
        if len(o) == 1:
            new_det[o] = v[0]
        else:
            new_det[list(o)] = list(v)
        dets.append(numpy.sort(new_det))
    return dets


def get_random_phmsd_opt(nup, ndown, nbasis, ndet=10, init=False, dist=None, cmplx_coeffs=True):
    if cmplx_coeffs:
        coeffs = numpy.random.rand(ndet) + 1j * numpy.random.rand(ndet)
    else:
        coeffs = numpy.random.rand(ndet) + 0j
    if init:
        a = numpy.random.rand(nbasis * (nup + ndown))
        if cmplx_coeffs:
            b = numpy.random.rand(nbasis * (nup + ndown))
            init_wfn = (a + 1j * b).reshape((nbasis, nup + ndown))
        else:
            init_wfn = a.reshape((nbasis, nup + ndown))
    else:
        init_wfn = None
    if dist is None:
        # want to evenly distribute determinants among N excitation levels
        ndet_level = max(int(ndet**0.5) // (int(nup**0.5)), 1)
        dist_a = [ndet_level] * (nup + 1)
        ndet_level = max(int(ndet**0.5) // (int(ndown**0.5)), 1)
        dist_b = [ndet_level] * (ndown + 1)
    else:
        assert len(dist) == 2
        dist_a, dist_b = dist
    d0a = numpy.array(numpy.arange(nup, dtype=numpy.int32))
    oa = [d0a]
    d0b = numpy.array(numpy.arange(ndown, dtype=numpy.int32))
    ob = [d0b]
    if ndet == 1:
        return (coeffs, numpy.array([d0a]), numpy.array([d0b])), init_wfn
    occ_a = numpy.arange(0, nup, dtype=numpy.int32)
    vir_a = numpy.arange(nup, nbasis, dtype=numpy.int32)
    occ_b = numpy.arange(0, ndown, dtype=numpy.int32)
    vir_b = numpy.arange(ndown, nbasis, dtype=numpy.int32)
    # dets = [(d0a, d0b)]
    dets = []
    for ialpha in range(0, nup + 1):
        oa = _gen_det_selection(d0a, vir_a, occ_a, dist_a, ialpha)
        if oa is None:
            continue
        for ibeta in range(0, ndown + 1):
            ob = _gen_det_selection(d0b, vir_b, occ_b, dist_b, ibeta)
            if ob is None:
                continue
            dets += list(itertools.product(oa, ob))
    occ_a, occ_b = zip(*dets)
    _ndet = min(len(occ_a), ndet)
    wfn = (coeffs, list(occ_a[:_ndet]), list(occ_b[:_ndet]))
    return wfn, init_wfn


def get_random_wavefunction(nelec, nbasis):
    na = nelec[0]
    nb = nelec[1]
    a = numpy.random.rand(nbasis * (na + nb))
    b = numpy.random.rand(nbasis * (na + nb))
    init = (a + 1j * b).reshape((nbasis, na + nb))
    return init


def generate_hamiltonian_low_mem(nmo, nelec, cplx=False):
    h1e = numpy.random.random((nmo, nmo))
    if cplx:
        h1e = h1e + 1j * numpy.random.random((nmo, nmo))
    chol = numpy.random.rand(nmo**3 * 4).reshape((nmo * 4, nmo, nmo))
    enuc = numpy.random.rand()
    return h1e, chol, enuc


def shaped_normal(shape, cmplx=False):
    size = numpy.prod(shape)
    if cmplx:
        arr_r = numpy.random.normal(size=size)
        arr_i = numpy.random.normal(size=size)
        arr = arr_r + 1j * arr_i
        arr = numpy.array(arr, dtype=numpy.complex128)
    else:
        arr = numpy.random.normal(size=size)
        arr = numpy.array(arr, dtype=numpy.float64)
    return arr.reshape(shape)


def get_random_sys_ham(nalpha, nbeta, nmo, naux, cmplx=False):
    sys = Generic(nelec=(nalpha, nbeta))
    chol = shaped_normal((naux, nmo, nmo), cmplx=cmplx)
    h1e = shaped_normal((nmo, nmo), cmplx=cmplx)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((naux, nmo * nmo)).T.copy(),
        # h1e_mod=h1e.copy(),
        ecore=0,
        verbose=False,
    )
    return sys, ham


def gen_random_test_instances(nmo, nocc, naux, nwalkers, seed=7, ndets=1):
    assert ndets == 1
    numpy.random.seed(seed)
    wfn = get_random_nomsd(nocc, nocc, nmo, ndet=ndets)
    h1e = shaped_normal((nmo, nmo))

    system = Generic(nelec=(nocc, nocc))
    chol = shaped_normal((naux, nmo, nmo))
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((naux, nmo * nmo)).T.copy(),
        ecore=0,
        verbose=False,
    )

    if ndets == 1:
        trial = SingleDet(wfn[1][0], (nocc, nocc), nmo)
    else:
        trial = NOCI(wfn, (nocc, nocc), nmo)
    walkers = UHFWalkersTrial(
        trial,
        wfn[1][0],
        system.nup,
        system.ndown,
        ham.nbasis,
        nwalkers,
        MPIHandler(),
    )
    walkers.build(trial)

    Ghalfa = shaped_normal((nwalkers, nocc, nmo), cmplx=True)
    Ghalfb = shaped_normal((nwalkers, nocc, nmo), cmplx=True)
    walkers.Ghalfa = Ghalfa
    walkers.Ghalfb = Ghalfb
    trial._rchola = shaped_normal((naux, nocc * nmo))
    trial._rcholb = shaped_normal((naux, nocc * nmo))
    trial._rH1a = shaped_normal((nocc, nmo))
    trial._rH1b = shaped_normal((nocc, nmo))
    return system, ham, walkers, trial


def build_random_phmsd_trial(
    num_elec: Tuple[int, int],
    num_basis: int,
    num_dets=1,
    wfn_type="opt",
    complex_trial: bool = False,
):
    _classes = {
        "naive": ParticleHoleNaive,
        "opt": ParticleHole,
        "chunked": ParticleHoleNonChunked,
        "slow": ParticleHoleSlow,
    }
    wfn, init = get_random_phmsd_opt(
        num_elec[0],
        num_elec[1],
        num_basis,
        ndet=num_dets,
        init=True,
        cmplx_coeffs=complex_trial,
    )
    return _classes[wfn_type](wfn, num_elec, num_basis), init


def build_random_noci_trial(
    num_elec: Tuple[int, int],
    num_basis: int,
    num_dets=1,
    complex_trial: bool = False,
):
    coeffs, wfn, init = get_random_nomsd(
        num_elec[0],
        num_elec[1],
        num_basis,
        ndet=num_dets,
        cplx=complex_trial,
        init=True,
    )
    trial = NOCI((coeffs, wfn), num_elec, num_basis)
    return trial, init


def build_random_single_det_trial(
    num_elec: Tuple[int, int],
    num_basis: int,
    complex_trial: bool = False,
    rhf_trial: bool = False,
):
    _, wfn, init = get_random_nomsd(
        num_elec[0], num_elec[1], num_basis, ndet=1, cplx=complex_trial, init=True
    )
    if rhf_trial:
        wfn[0, :, num_elec[0] :] = wfn[0, :, : num_elec[0]]
        init[:, num_elec[0] :] = init[:, : num_elec[0]]
    trial = SingleDet(wfn[0], num_elec, num_basis)
    return trial, init


def build_random_trial(
    num_elec: Tuple[int, int],
    num_basis: int,
    num_dets=1,
    trial_type="single_det",
    wfn_type="chunked",
    complex_trial: bool = False,
    rhf_trial: bool = False,
):
    if trial_type == "single_det":
        return build_random_single_det_trial(
            num_elec,
            num_basis,
            complex_trial=complex_trial,
            rhf_trial=rhf_trial,
        )
    elif trial_type == "noci":
        return build_random_noci_trial(num_elec, num_basis, complex_trial=complex_trial)
    elif trial_type == "phmsd":
        return build_random_phmsd_trial(
            num_elec,
            num_basis,
            complex_trial=complex_trial,
            wfn_type=wfn_type,
            num_dets=num_dets,
        )
    else:
        raise ValueError(f"Unkown trial type: {trial_type}")


@dataclass(frozen=True)
class TestData:
    trial: TrialWavefunctionBase
    walkers: BaseWalkers
    hamiltonian: HamGeneric
    propagator: PhaselessBase


def build_test_case_handlers_mpi(
    num_elec: Tuple[int, int],
    num_basis: int,
    mpi_handler: MPIHandler,
    num_dets=1,
    trial_type="phmsd",
    wfn_type="opt",
    complex_integrals: bool = False,
    complex_trial: bool = False,
    seed: Union[int, None] = None,
    rhf_trial: bool = False,
    two_body_only: bool = False,
    options: Union[dict, None] = None,
):
    if seed is not None:
        numpy.random.seed(seed)
    h1e, chol, _, _ = generate_hamiltonian(num_basis, num_elec, cplx=complex_integrals)
    system = Generic(nelec=num_elec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, num_basis**2)).T.copy(),
        ecore=0,
    )
    trial, init = build_random_trial(
        num_elec,
        num_basis,
        num_dets=num_dets,
        wfn_type=wfn_type,
        trial_type=trial_type,
        complex_trial=complex_trial,
        rhf_trial=rhf_trial,
    )
    trial.half_rotate(ham)
    trial.calculate_energy(system, ham)
    options["ntot_walkers"] = options.nwalkers * mpi_handler.comm.size
    # necessary for backwards compatabilty with tests
    if seed is not None:
        numpy.random.seed(seed)
    prop = PhaselessGeneric(time_step=options["dt"])
    prop.build(ham, trial)

    nwalkers = get_input_value(options, "nwalkers", default=10, alias=["num_walkers"])
    nsteps = get_input_value(options, "nsteps", default=25, alias=["num_steps"])
    pop_control = get_input_value(
        options, "population_control", default="pair_branch", alias=["pop_control"]
    )
    reconf_freq = get_input_value(options, "reconfiguration_freq", default=50)

    walkers = UHFWalkersTrial(
        trial, init, system.nup, system.ndown, ham.nbasis, nwalkers, MPIHandler()
    )
    walkers.build(trial)
    pcontrol = PopController(
        nwalkers, nsteps, mpi_handler, pop_control, reconfiguration_freq=reconf_freq
    )
    trial.calc_greens_function(walkers)
    for _ in range(options.num_steps):
        if two_body_only:
            prop.propagate_walkers_two_body(walkers, ham, trial)
        else:
            prop.propagate_walkers(walkers, ham, trial, trial.energy)
        walkers.reortho()
        pcontrol.pop_control(walkers, mpi_handler.comm)
        trial.calc_greens_function(walkers)

    return TestData(trial, walkers, ham, prop)


def build_test_case_handlers(
    num_elec: Tuple[int, int],
    num_basis: int,
    num_dets=1,
    trial_type="phmsd",
    wfn_type="opt",
    complex_integrals: bool = False,
    complex_trial: bool = False,
    seed: Union[int, None] = None,
    rhf_trial: bool = False,
    two_body_only: bool = False,
    choltol: float = 1e-3,
    reortho: bool = True,
    options: Union[dict, None] = None,
):
    if seed is not None:
        numpy.random.seed(seed)
    sym = 8
    if complex_integrals:
        sym = 4
    h1e, chol, _, eri = generate_hamiltonian(
        num_basis, num_elec, cplx=complex_integrals, sym=sym, tol=choltol
    )
    system = Generic(nelec=num_elec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, num_basis**2)).T.copy(),
        ecore=0,
    )
    ham.eri = eri.copy()
    trial, init = build_random_trial(
        num_elec,
        num_basis,
        num_dets=num_dets,
        wfn_type=wfn_type,
        trial_type=trial_type,
        complex_trial=complex_trial,
        rhf_trial=rhf_trial,
    )
    trial.half_rotate(ham)
    trial.calculate_energy(system, ham)
    # necessary for backwards compatabilty with tests
    if seed is not None:
        numpy.random.seed(seed)

    nwalkers = get_input_value(options, "nwalkers", default=10, alias=["num_walkers"])
    walkers = UHFWalkersTrial(
        trial, init, system.nup, system.ndown, ham.nbasis, nwalkers, MPIHandler()
    )
    walkers.build(trial)  # any intermediates that require information from trial

    prop = PhaselessGeneric(time_step=options["dt"])
    prop.build(ham, trial)

    trial.calc_greens_function(walkers)
    for _ in range(options.num_steps):
        if two_body_only:
            prop.propagate_walkers_two_body(walkers, ham, trial)
        else:
            prop.propagate_walkers(walkers, ham, trial, trial.energy)
        if reortho:
            walkers.reortho()
        trial.calc_greens_function(walkers)

    return TestData(trial, walkers, ham, prop)


def build_driver_test_instance(
    num_elec: Tuple[int, int],
    num_basis: int,
    num_dets=1,
    trial_type="phmsd",
    wfn_type="opt",
    complex_integrals: bool = False,
    complex_trial: bool = False,
    rhf_trial: bool = False,
    seed: Union[int, None] = None,
    density_diff=False,
    options: Union[dict, None] = None,
):
    if seed is not None:
        numpy.random.seed(seed)
    h1e, chol, _, _ = generate_hamiltonian(num_basis, num_elec, cplx=complex_integrals)
    system = Generic(nelec=num_elec)
    ham = HamGeneric(
        h1e=numpy.array([h1e, h1e]),
        chol=chol.reshape((-1, num_basis**2)).T.copy(),
        ecore=0,
    )
    if density_diff:
        ham.density_diff = True
    trial, _ = build_random_trial(
        num_elec,
        num_basis,
        num_dets=num_dets,
        wfn_type=wfn_type,
        trial_type=trial_type,
        complex_trial=complex_trial,
        rhf_trial=rhf_trial,
    )
    trial.half_rotate(ham)
    try:
        trial.calculate_energy(system, ham)
    except NotImplementedError:
        pass

    qmc_opts = get_input_value(options, "qmc", default={}, alias=["qmc_options"])
    qmc = QMCOpts(qmc_opts, verbose=0)
    qmc.nwalkers = qmc.nwalkers

    afqmc = AFQMC.build(
        num_elec,
        ham,
        trial,
        num_walkers=qmc.nwalkers,
        seed=qmc.rng_seed,
        num_steps_per_block=qmc.nsteps,
        num_blocks=qmc.nblocks,
        timestep=qmc.dt,
        stabilize_freq=qmc.nstblz,
        pop_control_freq=qmc.npop_control,
    )
    return afqmc
