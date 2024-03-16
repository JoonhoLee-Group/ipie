from ipie.estimators.generic import fock_generic

def fock_matrix(system, G):
    if system.name == "Generic":
        return fock_generic(system, G)
    else:
        print(f"# Fock matrix not implemented for {system.name}")
        return None
