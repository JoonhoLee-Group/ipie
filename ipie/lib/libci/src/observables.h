#ifndef _OBSERVABLES_H
#define _OBSERVABLES_H

#include <complex>
#include <vector>

#include "hamiltonian.h"
#include "wavefunction.h"

namespace ipie {
std::vector<ipie::complex_t> build_one_rdm(Wavefunction &wfn);
std::vector<ipie::complex_t> variational_energy(
    ipie::complex_t &ci_coeffs,
    std::vector<int> &occa,
    std::vector<int> &occb,
    std::vector<ipie::complex_t> &h1e,
    std::vector<ipie::complex_t> &h2e,
    ipie::complex_t e0);
std::vector<ipie::complex_t> variational_energy(Wavefunction &wfn, Hamiltonian &ham);
}  // namespace ipie
#endif