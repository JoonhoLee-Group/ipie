#ifndef _OBSERVABLES_H
#define _OBSERVABLES_H

#include <complex>
#include <vector>

#include "hamiltonian.h"
#include "wavefunction.h"

namespace ipie {
std::vector<ipie::complex_t> build_one_rdm(Wavefunction &wfn);
energy_t compute_variational_energy(Wavefunction &wfn, Hamiltonian &ham);
}  // namespace ipie
#endif