from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.propagation.phaseless_generic import PhaselessGeneric
from ipie.addons.eph.propagation.holstein import HolsteinPropagator
from ipie.addons.eph.hamiltonians.holstein import HolsteinModel

Propagator = {GenericRealChol: PhaselessGeneric, GenericComplexChol: PhaselessGeneric, HolsteinModel: HolsteinPropagator}
