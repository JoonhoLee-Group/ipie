#ifndef _OBSERVABLES_H
#define _OBSERVABLES_H

#include <complex>
#include <vector>

#include "hamiltonian.h"
#include "wavefunction.h"

namespace ipie {
std::vector<std::complex<double>> one_rdm(Wavefunction &wfn);
std::vector<std::complex<double>> variational_energy(Wavefunction &wfn, Hamiltonian &ham);
}  // namespace ipie
#endif