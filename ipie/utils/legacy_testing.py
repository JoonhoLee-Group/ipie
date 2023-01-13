import copy

from ipie.legacy.propagation.continuous import Continuous as LegacyContinuous
from ipie.legacy.walkers.single_det import SingleDetWalker
from ipie.legacy.trial_wavefunction.multi_slater import MultiSlater
from ipie.legacy.estimators.local_energy import local_energy_generic_cholesky_opt
from ipie.utils.misc import dotdict


def build_legacy_test_case(
    wfn, init, system, ham, num_steps, num_walkers, dt, nstblz=5
):
    qmc = dotdict({"dt": dt, "nstblz": nstblz})
    options = {"hybrid": True}
    ham_legacy = copy.deepcopy(ham)
    ham_legacy.control_variate = False
    trial = MultiSlater(system, ham_legacy, wfn, init=init)
    trial.half_rotate(system, ham_legacy)
    if trial.ndets == 1:
        trial.psi = trial.psi[0]
    prop = LegacyContinuous(system, ham_legacy, trial, qmc, options=options)

    walkers = [SingleDetWalker(system, ham_legacy, trial) for iw in range(num_walkers)]
    for i in range(num_steps):
        for walker in walkers:
            prop.propagate_walker(walker, system, ham_legacy, trial, 0.0)
            _ = walker.reortho(trial)  # reorthogonalizing to stablize
            walker.greens_function(trial)
    return walkers


def get_legacy_walker_energies(system, ham, trial, walkers):
    etots = []
    e1s = []
    e2s = []
    for iw, walker in enumerate(walkers):
        e = local_energy_generic_cholesky_opt(
            system,
            ham,
            walker.G[0],
            walker.G[1],
            walker.Ghalf[0],
            walker.Ghalf[1],
            trial._rchola,
            trial._rcholb,
        )
        etots += [e[0]]
        e1s += [e[1]]
        e2s += [e[2]]
    return etots, e1s, e2s
