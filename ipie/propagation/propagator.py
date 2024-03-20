from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.propagation.phaseless_generic import PhaselessGeneric
from ipie.addons.propagator import PropagatorAddons

Propagator = {GenericRealChol: PhaselessGeneric, GenericComplexChol: PhaselessGeneric}
Propagator.update(PropagatorAddons)
