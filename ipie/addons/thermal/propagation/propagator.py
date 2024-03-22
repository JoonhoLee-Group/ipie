from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.addons.thermal.propagation.phaseless_generic import PhaselessGeneric

Propagator = {GenericRealChol: PhaselessGeneric, GenericComplexChol: PhaselessGeneric}
