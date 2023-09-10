#ifndef _OBSERVABLES_H
#define _OBSERVABLES_H

#include <complex>
#include <vector>

#include "hamiltonian.h"
#include "wavefunction.h"

namespace ipie {
energy_t compute_variational_energy(Wavefunction &wfn, Hamiltonian &ham);
}  // namespace ipie
#endif