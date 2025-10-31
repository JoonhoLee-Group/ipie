import numpy

from ipie.systems import Generic

from ipie.addons.eph.hamiltonians.holstein import HolsteinModel
from ipie.addons.eph.walkers.eph_walkers import EPhWalkers
from ipie.addons.eph.trial_wavefunction.toyozawa import ToyozawaTrial
from ipie.addons.eph.trial_wavefunction.coherent_state import CoherentStateTrial


def get_random_sys_holstein(nelec, nbasis, pbc):
    sys = Generic(nelec=nelec)
    g = numpy.random.rand()
    t = numpy.random.rand()
    w0 = numpy.random.rand()
    ham = HolsteinModel(g=g, t=t, w0=w0, nsites=nbasis, pbc=pbc)
    ham.build()
    return sys, ham


def get_random_wavefunction(nelec, nbasis):
    init = numpy.random.random((nbasis, (nelec[0] + nelec[1] + 1)))
    return init


def build_random_toyozawa_trial(nelec, nbasis, w0):
    wfn = get_random_wavefunction(nelec, nbasis)
    trial = ToyozawaTrial(wavefunction=wfn, w0=w0, num_elec=nelec, num_basis=nbasis)
    return trial


def build_random_coherent_state_trial(nelec, nbasis, w0):
    wfn = get_random_wavefunction(nelec, nbasis)
    trial = CoherentStateTrial(wavefunction=wfn, w0=w0, num_elec=nelec, num_basis=nbasis)
    return trial


def build_random_trial(nelec, nbasis, w0, trial_type):
    if trial_type == "coherent_state":
        return build_random_coherent_state_trial(nelec, nbasis, w0)
    elif trial_type == "toyozawa":
        return build_random_toyozawa_trial(nelec, nbasis, w0)
    else:
        raise ValueError(f"Unkown trial type: {trial_type}")


def gen_random_test_instances(nelec, nbasis, nwalkers, trial_type, seed=7):
    numpy.random.seed(seed)

    wfn = get_random_wavefunction(nelec, nbasis)
    sys, ham = get_random_sys_holstein(nelec, nbasis, True)
    trial = build_random_trial(nelec, nbasis, ham.w0, trial_type)
    walkers = EPhWalkers(wfn, nelec[0], nelec[1], nbasis, nwalkers)
    walkers.build(trial)

    return sys, ham, walkers, trial
