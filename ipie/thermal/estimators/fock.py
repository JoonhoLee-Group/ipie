from ipie.estimators.generic import fock_generic
from ipie.thermal.estimators.ueg import fock_ueg


def fock_matrix(system, G):
    if system.name == "UEG":
        return fock_ueg(system, G)
    elif system.name == "Generic":
        return fock_generic(system, G)
    elif system.name == "Hubbard":
        raise NotImplementedError
    else:
        print(f"# Fock matrix not implemented for {system.name}")
        return None
