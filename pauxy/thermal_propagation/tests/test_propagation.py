import numpy
import pytest
from pauxy.systems.hubbard import Hubbard
from pauxy.estimators.thermal import greens_function, one_rdm_from_G, particle_number
from pauxy.trial_density_matrices.onebody import OneBody
from pauxy.trial_density_matrices.mean_field import MeanField
from pauxy.thermal_propagation.hubbard import ThermalDiscrete
from pauxy.walkers.thermal import ThermalWalker
from pauxy.utils.misc import dotdict, update_stack

@pytest.mark.unit
def test_hubbard():
    options = {'nx': 4, 'ny': 4, 'U': 4, 'mu': 1.0, 'nup': 7, 'ndown': 7}
    system = Hubbard(options, verbose=False)
    beta = 2.0
    dt = 0.05
    nslice = int(round(beta/dt))
    trial = OneBody(system, beta, dt)
    numpy.random.seed(7)
    qmc = dotdict({'dt': dt, 'nstblz': 10})
    prop = ThermalDiscrete(system, trial, qmc, verbose=False)
    walker1 = ThermalWalker(system, trial,
                            walker_opts={'stack_size': 1, 'low_rank': False},
                            verbose=False)
    for ts in range(0,nslice):
        prop.propagate_walker(system, walker1, ts, 0)
        walker1.weight /= 1.0e6
    numpy.random.seed(7)
    walker2 = ThermalWalker(system, trial,
                            walker_opts={'stack_size': 10, 'low_rank': False},
                            verbose=False)
    energies = []
    for ts in range(0,nslice):
        prop.propagate_walker(system, walker2, ts, 0)
        walker2.weight /= 1.0e6
        # if ts % 10 == 0:
            # energies.append(walker2.local_energy(system)[0])
    # import matplotlib.pyplot as pl
    # pl.plot(energies, markersize=2)
    # pl.show()
    assert walker1.weight == pytest.approx(walker2.weight)
    assert numpy.linalg.norm(walker1.G-walker2.G) == pytest.approx(0)
    assert walker1.local_energy(system)[0] == pytest.approx(walker2.local_energy(system)[0])

@pytest.mark.unit
def test_propagate_walker():
    options = {'nx': 4, 'ny': 4, 'U': 4, 'mu': 1.0, 'nup': 7, 'ndown': 7}
    system = Hubbard(options, verbose=False)
    beta = 2.0
    dt = 0.05
    nslice = int(round(beta/dt))
    trial = OneBody(system, beta, dt)
    numpy.random.seed(7)
    qmc = dotdict({'dt': dt, 'nstblz': 1})
    prop = ThermalDiscrete(system, trial, qmc, verbose=False)
    walker1 = ThermalWalker(system, trial,
                            walker_opts={'stack_size': 1, 'low_rank': False},
                            verbose=False)
    walker2 = ThermalWalker(system, trial,
                            walker_opts={'stack_size': 1, 'low_rank': False},
                            verbose=False)
    rands = numpy.random.random(system.nbasis)
    I = numpy.eye(system.nbasis)
    BV = numpy.zeros((2,system.nbasis))
    BV[0] = 1.0
    BV[1] = 1.0
    walker2.greens_function(trial, slice_ix=0)
    walker1.greens_function(trial, slice_ix=0)
    for it in range(0,nslice):
        rands = numpy.random.random(system.nbasis)
        BV = numpy.zeros((2,system.nbasis))
        BV[0] = 1.0
        BV[1] = 1.0
        for i in range(system.nbasis):
            if rands[i] > 0.5:
                xi = 0
            else:
                xi = 1
            BV[0,i] = prop.auxf[xi,0]
            BV[1,i] = prop.auxf[xi,1]
            # Check overlap ratio
            if it % 20 == 0:
                probs1 = prop.calculate_overlap_ratio(walker1,i)
                G2old = walker2.greens_function(trial, slice_ix=it, inplace=False)
                B = numpy.einsum('ki,kij->kij', BV, prop.BH1)
                walker2.stack.stack[it] = B
                walker2.greens_function(trial, slice_ix=it)
                G2 = walker2.G
                pdirect = numpy.linalg.det(G2old[0])/numpy.linalg.det(G2[0])
                pdirect *= 0.5*numpy.linalg.det(G2old[1])/numpy.linalg.det(G2[1])
                pdirect == pytest.approx(probs1[xi])
            prop.update_greens_function(walker1, i, xi)
        B = numpy.einsum('ki,kij->kij', BV, prop.BH1)
        walker1.stack.update(B)
        if it % prop.nstblz == 0:
            walker1.greens_function(None,
                                    walker1.stack.time_slice-1)
        walker2.stack.stack[it] = B
        walker2.greens_function(trial, slice_ix=it)
        numpy.linalg.norm(walker1.G-walker2.G) == pytest.approx(0.0)
        prop.propagate_greens_function(walker1)

@pytest.mark.unit
def test_propagate_walker_free():
    options = {'nx': 4, 'ny': 4, 'U': 4, 'mu': 1.0, 'nup': 8, 'ndown': 8}
    system = Hubbard(options, verbose=False)
    beta = 0.5
    dt = 0.05
    nslice = int(round(beta/dt))
    trial = MeanField(system, beta, dt, verbose=False)
    numpy.random.seed(7)
    qmc = dotdict({'dt': dt, 'nstblz': 1})
    prop = ThermalDiscrete(system, trial, qmc, {'charge_decomposition': False, 'free_projection': True}, verbose=False)
    walker = ThermalWalker(system, trial,
                           walker_opts={'stack_size': 1, 'low_rank': False},
                           verbose=True)
    for ts in range(0,nslice):
        prop.propagate_walker(system, walker, ts, 0)
        walker.greens_function(None)
        rdm = one_rdm_from_G(walker.G)

@pytest.mark.unit
def test_update_gf():
    options = {'nx': 4, 'ny': 4, 'U': 4, 'mu': 1.0, 'nup': 8, 'ndown': 8}
    system = Hubbard(options, verbose=False)
    beta = 2.0
    dt = 0.05
    nslice = int(round(beta/dt))
    trial = OneBody(system, beta, dt)
    numpy.random.seed(7)
    qmc = dotdict({'dt': dt, 'nstblz': 1})
    prop = ThermalDiscrete(system, trial, qmc, verbose=False)
    walker = ThermalWalker(system, trial,
                            walker_opts={'stack_size': 1, 'low_rank': False},
                            verbose=False)
    rands = numpy.random.random(system.nbasis)
    I = numpy.eye(system.nbasis)
    BV = numpy.zeros((2,system.nbasis))
    BV[0] = 1.0
    BV[1] = 1.0
    walker.greens_function(trial, slice_ix=0)
    rands = numpy.random.random(system.nbasis)
    BV = numpy.zeros((2,system.nbasis))
    BV[0] = 1.0
    BV[1] = 1.0
    for i in range(system.nbasis):
        if rands[i] > 0.5:
            xi = 0
        else:
            xi = 1
        BV[0,i] = prop.auxf[xi,0]
        BV[1,i] = prop.auxf[xi,1]
        prop.update_greens_function(walker, i, xi)
    G = walker.G.copy()
    B = numpy.einsum('ki,kij->kij', BV, prop.BH1)
    walker.stack.update(B)
    Gnew = walker.greens_function(trial, slice_ix=0, inplace=False)
    assert numpy.linalg.norm(G-Gnew) == pytest.approx(0.0)
    prop.propagate_greens_function(walker)
    Gnew = walker.greens_function(trial, slice_ix=1, inplace=False)
    assert numpy.linalg.norm(walker.G-Gnew) == pytest.approx(0.0)
