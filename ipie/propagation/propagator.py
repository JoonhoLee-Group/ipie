from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.hamiltonians.elph.holstein import HolsteinModel
from ipie.propagation.phaseless_generic import PhaselessGeneric
from ipie.propagation.holstein import HolsteinPropagatorImportance

Propagator = {GenericRealChol: PhaselessGeneric, GenericComplexChol: PhaselessGeneric, HolsteinModel: HolsteinPropagatorImportance}
