# Author: Maxine Luo <man.luo@mpq.mpg.de>,
#         Victor Chen <victor.chen@tum.de>
#
import pytest

from ipie.addons.ithc.propagation.phaseless_ithc import PhaselessITHC
from ipie.addons.ithc.utils.testing import gen_random_test_instances
from ipie.addons.ithc.utils.testing import gen_random_test_instances_ithc


@pytest.mark.unit
def test_extended_propagator():
    nmo = 10
    nocc = 8
    naux = 30
    nwalker = 10
    dt = 0.001
    system, ham, walker_batch, trial = gen_random_test_instances_ithc(nmo, nocc, naux, nwalker)
    trial.half_rotate(ham)
    propagation = PhaselessITHC(dt)
    propagation.build(ham, trial, walker_batch)
    propagation.propagate_walkers_two_body_first_order(walker_batch, ham, trial)
    propagation.propagate_walkers(walker_batch, ham, trial, eshift=0.1)
    return


if __name__ == "__main__":
    test_extended_propagator()
