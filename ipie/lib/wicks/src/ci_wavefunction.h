#ifndef _CI_WAVEFUNCTION_H
#define _CI_WAVEFUNCTION_H

#include <complex>
#include <vector>
#include "bitstring.h"

namespace ipie {

struct CIWavefunction {
    std::vector<std::complex<double>> coeffs;
    std::vector<BitString> dets;

    CIWavefunction(std::vector<std::complex<double>> &ci_coeffs, std::vector<BitString> &dets);
    CIWavefunction build_ci_wavefunction(
        std::vector<std::complex<double>> &ci_coeffs, std::vector<int> &occa, std::vector<int> &occb);

    std::vector<std::complex<double>> build_one_rdm(size_t num_dets_to_use);
    std::vector<std::complex<double>> compute_variational_energy(size_t num_dets_to_use);
};

}  // namespace ipie

#endif