from ipie.thermal.estimators.generic import fock_generic
from ipie.thermal.estimators.ueg import fock_ueg


def fock_matrix(hamiltonian, G):
    if hamiltonian.name == "UEG":
        return fock_ueg(hamiltonian, G)
    elif hamiltonian.name == "Generic":
        return fock_generic(hamiltonian, G)
    else:
        print(f"# Fock matrix not implemented for {hamiltonian.name}")
        return None
