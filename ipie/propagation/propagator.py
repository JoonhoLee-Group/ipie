from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol, GenericRealCholChunked
from ipie.propagation.phaseless_generic import PhaselessGeneric, PhaselessGenericChunked

# Propagator = {GenericRealChol: PhaselessGeneric, GenericComplexChol: PhaselessGeneric}
Propagator = {GenericRealChol: PhaselessGenericChunked, GenericComplexChol: PhaselessGenericChunked, GenericRealCholChunked: PhaselessGenericChunked}
