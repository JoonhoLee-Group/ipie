from ipie.hamiltonians.generic import GenericRealChol, GenericComplexChol
from ipie.hamiltonians.isdf import GenericRealISDF
from ipie.hamiltonians.kpt_hamiltonian import KptComplexChol, KptComplexCholSymm
from ipie.hamiltonians.kpt_isdf_hamiltonian import KptISDF
from ipie.hamiltonians.generic_chunked import GenericRealCholChunked
from ipie.hamiltonians.chunked_isdf import GenericRealISDFChunked
from ipie.hamiltonians.kpt_chunked import KptComplexCholChunked
from ipie.propagation.phaseless_generic import (
    PhaselessGeneric,
    PhaselessGenericChunked,
)
from ipie.propagation.phaseless_isdf import (
    PhaselessISDF,
    PhaselessISDFChunked,
)
from ipie.propagation.phaseless_kpt import (
    PhaselessKptChol,
    PhaselessKptCholChunked,
    PhaselessKptISDF,
)

# Propagator = {GenericRealChol: PhaselessGeneric, GenericComplexChol: PhaselessGeneric}
Propagator = {
    GenericRealChol: PhaselessGeneric,
    GenericComplexChol: PhaselessGeneric,
    GenericRealCholChunked: PhaselessGenericChunked,
    GenericRealISDF: PhaselessISDF,
    GenericRealISDFChunked: PhaselessISDFChunked,
    KptComplexChol: PhaselessKptChol,
    KptComplexCholSymm: PhaselessKptChol,
    KptComplexCholChunked: PhaselessKptCholChunked,
    KptISDF: PhaselessKptISDF,
}
