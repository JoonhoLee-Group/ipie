#ifndef _CIWAVEFUNCTION_H
#define _CIWAVEFUNCTION_H

#include <complex>
#include <vector>

#include "bitstring.h"

namespace ipie {

class CIWavefunction {
   public:
    CIWavefunction(std::vector<std::complex<double>> &ci_coeffs, std::vector<BitString> &determinants);
    // factory method
    CIWavefunction build_ci_wavefunction(
        std::vector<std::complex<double>> &ci_coeffs, std::vector<int> &occa, std::vector<int> &occb);

    std::vector<std::complex<double>> build_one_rdm(size_t num_dets_to_use);
    std::vector<std::complex<double>> compute_variational_energy(size_t num_dets_to_use);

   private:
    std::vector<std::complex<double>> coeffs;
    std::vector<BitString> dets;
    size_t num_dets;
};
}  // namespace ipie

#endif
