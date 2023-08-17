import numpy
from afqmctools.utils.qmcpack_utils import write_xml_input

from ipie.config import MPI
from ipie.legacy.hamiltonians.hubbard import Hubbard
from ipie.trial_wavefunction.utils import get_trial_wavefunction
from ipie.utils.io import write_qmcpack_dense, write_qmcpack_wfn
from ipie.utils.linalg import modified_cholesky

comm = MPI.COMM_WORLD

system = Hubbard({"nup": 8, "ndown": 8, "nx": 4, "ny": 4, "U": 8.0})

nb = system.nbasis
na = system.nup
nel = sum(system.nelec)
hcore = system.T[0]
eris = numpy.zeros((nb, nb, nb, nb))
for i in range(nb):
    eris[i, i, i, i] = system.U
eris = eris.reshape(nb * nb, nb * nb)
chol = modified_cholesky(eris)
write_qmcpack_dense(hcore, chol.T.copy(), system.nelec, system.nbasis)
options = {"name": "UHF", "spin_proj": True, "ninitial": 20}
trial = get_trial_wavefunction(system, options=options, verbose=True, comm=comm)
init = [trial.init[:, :na].copy(), trial.init[:, na:].copy()]
write_qmcpack_wfn(
    "wfn.h5",
    (numpy.array([1.0 + 0j]), trial.psi.reshape(1, nb, nel)),
    "uhf",
    system.nelec,
    system.nbasis,
    init=init,
)
write_xml_input("input.xml", "hamiltonian.h5", "wfn.h5")
