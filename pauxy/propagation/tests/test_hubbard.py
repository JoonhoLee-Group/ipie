import numpy
import pytest
from pauxy.systems.hubbard import Hubbard, decode_basis
from pauxy.propagation.hubbard import Hirsch
from pauxy.propagation.continuous import Continuous
from pauxy.trial_wavefunction.multi_slater import MultiSlater
from pauxy.trial_wavefunction.uhf import UHF
from pauxy.walkers.handler import Walkers
from pauxy.walkers.single_det import SingleDetWalker
from pauxy.utils.misc import dotdict
from pauxy.estimators.greens_function import gab

options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
system = Hubbard(inputs=options)
eigs, eigv = numpy.linalg.eigh(system.H1[0])
coeffs = numpy.array([1.0+0j])
wfn = numpy.zeros((1,system.nbasis,system.ne), dtype=numpy.complex128)
wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
trial = MultiSlater(system, (coeffs, wfn))
trial.psi = trial.psi[0]

@pytest.mark.unit
def test_hubbard_spin():
    qmc = dotdict({'dt': 0.01, 'nstblz': 5})
    prop = Hirsch(system, trial, qmc)
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    numpy.random.seed(7)
    nup = system.nup
    prop.propagate_walker_constrained(walker, system, trial, 0.0)
    walker_ref = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    ovlpa = numpy.linalg.det(numpy.dot(trial.psi[:,:nup].conj().T, walker_ref.phi[:,:nup]))
    assert ovlpa == pytest.approx(walker_ref.ovlp)
    # Alpha electrons
    walker_ref.phi[:,:nup] = numpy.dot(prop.bt2[0], walker_ref.phi[:,:nup])
    BV = numpy.diag([prop.auxf[int(x.real),0] for x in walker.field_configs.configs[0]])
    walker_ref.phi[:,:nup] = numpy.dot(BV, walker_ref.phi[:,:nup])
    walker_ref.phi[:,:nup] = numpy.dot(prop.bt2[0], walker_ref.phi[:,:nup])
    numpy.testing.assert_allclose(walker.phi[:,:nup], walker_ref.phi[:,:nup], atol=1e-14)
    ovlpa = numpy.linalg.det(numpy.dot(trial.psi[:,:nup].conj().T, walker_ref.phi[:,:nup]))
    # Beta electrons
    BV = numpy.diag([prop.auxf[int(x.real),1] for x in walker.field_configs.configs[0]])
    walker_ref.phi[:,nup:] = numpy.dot(prop.bt2[1], walker_ref.phi[:,nup:])
    walker_ref.phi[:,nup:] = numpy.dot(BV, walker_ref.phi[:,nup:])
    walker_ref.phi[:,nup:] = numpy.dot(prop.bt2[1], walker_ref.phi[:,nup:])
    numpy.testing.assert_allclose(walker.phi[:,nup:], walker_ref.phi[:,nup:], atol=1e-14)
    # Test overlap
    ovlpb = numpy.linalg.det(numpy.dot(trial.psi[:,nup:].conj().T, walker_ref.phi[:,nup:]))
    ovlp_ = walker.calc_overlap(trial)
    assert walker.ot == pytest.approx(ovlpa*ovlpb)


@pytest.mark.unit
def test_update_greens_function():
    qmc = dotdict({'dt': 0.01, 'nstblz': 5})
    prop = Hirsch(system, trial, qmc)
    walker = SingleDetWalker(system, trial)
    numpy.random.seed(7)
    prop.kinetic_importance_sampling(walker, system, trial)
    delta = prop.delta
    nup = system.nup
    soffset = walker.phi.shape[0] - system.nbasis
    fields = [1 if numpy.random.random() > 0.5 else 0 for i in range(system.nbasis)]
    # Reference values
    bvu = numpy.diag([prop.auxf[x][0] for x in fields])
    bvd = numpy.diag([prop.auxf[x][1] for x in fields])
    pnu = numpy.dot(bvu, walker.phi[:,:system.nup])
    pnd = numpy.dot(bvd, walker.phi[:,system.nup:])
    gu = gab(trial.psi[:,:system.nup], pnu)
    gd = gab(trial.psi[:,system.nup:], pnd)
    nup = system.nup
    ovlp = numpy.dot(trial.psi[:,:system.nup].conj().T, walker.phi[:,nup:])
    for i in range(system.nbasis):
        vtup = walker.phi[i,:nup] * delta[fields[i],0]
        vtdn = walker.phi[i,nup:] * delta[fields[i],1]
        walker.phi[i,:nup] = walker.phi[i,:nup] + vtup
        walker.phi[i,nup:] = walker.phi[i,nup:] + vtdn
        walker.update_inverse_overlap(trial, vtup, vtdn, i)
        prop.update_greens_function(walker, trial, i, nup)
        guu = gab(trial.psi[:,:system.nup], walker.phi[:,:nup])
        gdd = gab(trial.psi[:,system.nup:], walker.phi[:,nup:])
        assert guu[i,i] == pytest.approx(walker.G[0,i,i])
        assert gdd[i,i] == pytest.approx(walker.G[1,i,i])

@pytest.mark.unit
def test_hubbard_charge():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Hubbard(inputs=options)
    wfn = numpy.zeros((1,system.nbasis,system.ne), dtype=numpy.complex128)
    count = 0
    uhf = UHF(system, {'ueff': 4.0, 'initial': 'checkerboard'}, verbose=True)
    wfn[0] = uhf.psi.copy()
    trial = MultiSlater(system, (coeffs, wfn))
    trial.psi = trial.psi[0]
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    qmc = dotdict({'dt': 0.01, 'nstblz': 5})
    options = {'charge_decomposition': True}
    prop = Hirsch(system, trial, qmc, options=options, verbose=True)
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    numpy.random.seed(7)
    nup = system.nup
    # prop.propagate_walker_constrained(walker, system, trial, 0.0)
    prop.two_body(walker, system, trial)
    walker_ref = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    # Alpha electrons
    BV = numpy.diag([prop.auxf[int(x.real),0] for x in walker.field_configs.configs[0]])
    ovlp = walker_ref.calc_overlap(trial)
    walker_ref.phi[:,:nup] = numpy.dot(BV, walker_ref.phi[:,:nup])
    walker_ref.phi[:,nup:] = numpy.dot(BV, walker_ref.phi[:,nup:])
    ovlp *= walker_ref.calc_overlap(trial)
    assert ovlp != pytest.approx(walker.ot)
    for i in walker.field_configs.configs[0]:
        ovlp *= prop.aux_wfac[int(i.real)]
    assert ovlp.imag == pytest.approx(0.0, abs=1e-10)
    assert ovlp == pytest.approx(walker.ot)

@pytest.mark.unit
def test_hubbard_continuous_spin():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Hubbard(inputs=options)
    wfn = numpy.zeros((1,system.nbasis,system.ne), dtype=numpy.complex128)
    count = 0
    numpy.random.seed(7)
    uhf = UHF(system, {'ueff': 4.0}, verbose=True)
    wfn[0] = uhf.psi.copy()
    trial = MultiSlater(system, (coeffs, wfn))
    trial.psi = trial.psi[0]
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    qmc = dotdict({'dt': 0.01, 'nstblz': 5})
    options = {'charge_decomposition': False}
    prop = Continuous(system, trial, qmc, options=options, verbose=True)
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    nup = system.nup
    prop.propagate_walker(walker, system, trial, 0.0)
    assert walker.ovlp.imag == pytest.approx(0.0)
    assert walker.ovlp.real == pytest.approx(0.765551499039435)

@pytest.mark.unit
def test_hubbard_discrete_fp():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Hubbard(inputs=options)
    numpy.random.seed(7)
    wfn = numpy.random.random(system.nbasis*system.ne).reshape((1,system.nbasis,system.ne))
    trial = MultiSlater(system, (coeffs, wfn))
    trial.psi = trial.psi[0]
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    qmc = dotdict({'dt': 0.01, 'nstblz': 5})
    options = {'free_projection': True}
    prop = Hirsch(system, trial, qmc, options=options, verbose=True)
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    nup = system.nup
    numpy.random.seed(7)
    prop.propagate_walker(walker, system, trial, 0.0)
    ovlp_a = walker.greens_function(trial)
    ovlp = walker.calc_overlap(trial)
    e1 = walker.weight*walker.phase*ovlp*walker.local_energy(system, trial)[0]
    walker = SingleDetWalker(system, trial, nbp=1, nprop_tot=1)
    detR = walker.reortho(trial)
    walker.weight *= detR
    numpy.random.seed(7)
    prop.propagate_walker(walker, system, trial, 0.0)
    ovlp = walker.greens_function(trial)
    e2 = walker.weight*walker.phase*ovlp*walker.local_energy(system, trial)[0]
    assert e1 == pytest.approx(e2)

@pytest.mark.unit
def test_hubbard_diff():
    options = {'nx': 4, 'ny': 4, 'nup': 8, 'ndown': 8, 'U': 4}
    system = Hubbard(inputs=options)
    numpy.random.seed(7)
    # eigenvalue decomp is sensitive to machine.
    wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
    wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
    trial = MultiSlater(system, (coeffs, wfn))
    qmc = dotdict({'dt': 0.01, 'nstblz': 5, 'nwalkers': 10})
    options = {'free_projection': True}
    walkers = Walkers(system, trial, qmc)
    # Discrete
    prop = Hirsch(system, trial, qmc, options=options, verbose=True)
    for w in walkers.walkers:
        for i in range(0,10):
            prop.propagate_walker(w, system, trial, 0.0)
        detR = w.reortho(trial)
    # Continuous
    trial = MultiSlater(system, (coeffs, wfn))
    walkers2 = Walkers(system, trial, qmc, nbp=1, nprop_tot=1)
    prop = Continuous(system, trial, qmc, options=options, verbose=True)
    for w in walkers2.walkers:
        for i in range(0,10):
            prop.propagate_walker(w, system, trial, 0.0)
        detR2 = w.reortho(trial)
    # assert abs(w.ovlp) == pytest.approx(0.30079734335907987)
    # assert detR2 == pytest.approx(0.25934872296001504)
    # CP
    trial = MultiSlater(system, (coeffs, wfn))
    walkers2 = Walkers(system, trial, qmc)
    options = {'free_projection': False}
    prop = Hirsch(system, trial, qmc, options=options, verbose=True)
    for w in walkers2.walkers:
        for i in range(0,10):
            prop.propagate_walker(w, system, trial, 0.0)
        detR2 = w.reortho(trial)
    # assert detR2 == pytest.approx(13.678898660838367)

def total_energy(walkers, system, trial, fp=False):
    num = 0.0
    den = 0.0
    for w in walkers:
        w.greens_function(trial)
        e = w.local_energy(system)[0]
        if fp:
            num += w.weight*w.ot*e*numpy.exp(w.log_detR-w.log_detR_shift)
            den += w.weight*w.ot*numpy.exp(w.log_detR-w.log_detR_shift)
        else:
            num += w.weight*e
            den += w.weight
    return num/den


@pytest.mark.unit
def test_hubbard_ortho():
    options = {'nx': 4, 'ny': 4, 'nup': 5, 'ndown': 5, 'U': 4}
    system = Hubbard(inputs=options)
    numpy.random.seed(7)
    wfn = numpy.zeros((1,system.nbasis,system.ne), dtype=numpy.complex128)
    wfn[0,:,:system.nup] = eigv[:,:system.nup].copy()
    wfn[0,:,system.nup:] = eigv[:,:system.ndown].copy()
    trial = MultiSlater(system, (coeffs, wfn))
    qmc = dotdict({'dt': 0.01, 'nstblz': 5, 'nwalkers': 10, 'ntot_walkers': 10})
    options = {'free_projection': True}
    wopt = {'population_control': 'pair_branch', 'use_log_shift': True}
    walkers = Walkers(system, trial, qmc, walker_opts=wopt)
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    # Discrete
    prop = Hirsch(system, trial, qmc, options=options, verbose=True)
    energies = []
    for i in range(0,100):
        if i % 5 == 0:
            walkers.orthogonalise(trial, free_projection=True)
        for w in walkers.walkers:
            prop.propagate_walker(w, system, trial, 0.0)
        # print(w.detR_shift, w.log_detR_shift,  w.log_detR, w.detR)
        if i % 1 == 0:
            energies.append(total_energy(walkers.walkers, system, trial, True).real)
        if i % 5 == 0:
            # print([w.local_energy(system)[0].real for w in walkers.walkers])
            # print([(w.weight).real for w in walkers.walkers])
            walkers.pop_control(comm)
            # print([w.local_energy(system)[0].real for w in walkers.walkers])
            # print([(w.weight).real for w in walkers.walkers])
    # CP
    trial = MultiSlater(system, (coeffs, wfn))
    walkers2 = Walkers(system, trial, qmc)
    options = {'free_projection': False}
    prop = Hirsch(system, trial, qmc, options=options, verbose=True)
    numpy.random.seed(7)
    energies2 = []
    for i in range(0,100):
        if i % 5 == 0:
            walkers2.orthogonalise(trial, free_projection=False)
        for w in walkers2.walkers:
            prop.propagate_walker(w, system, trial, 0.0)
        if i % 1 == 0:
            energies2.append(total_energy(walkers2.walkers, system, trial, False).real)
        if i % 5 == 0:
            walkers2.pop_control(comm)
    for w in walkers2.walkers:
        w.greens_function(trial)
    weights_2 = [w.weight for w in walkers2.walkers]
    # print(weights, weights_2)
    # energies_2 = [w.local_energy(system)[0].real for w in walkers2.walkers]
    # import matplotlib.pyplot as pl
    # print(energies,energies2)
    # pl.plot(energies, label='fp', linestyle=':')
    # pl.plot(energies2, label='cpmc', linestyle='-.')
    # pl.axhline(-1.22368*16)
    # pl.ylim([-25,-17])
    # pl.legend()
    # pl.show()

