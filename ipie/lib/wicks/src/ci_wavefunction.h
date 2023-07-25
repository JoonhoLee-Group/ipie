#ifndef _CIWAVEFUNCTION_H
#define _CIWAVEFUNCTION_H

#include <complex>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"

namespace ipie {

class CIWavefunction {
   public:
    CIWavefunction(std::vector<std::complex<double>> &ci_coeffs, std::vector<BitString> &determinants);
    // factory method
    CIWavefunction build_ci_wavefunction(
        std::vector<std::complex<double>> &ci_coeffs, std::vector<int> &occa, std::vector<int> &occb);

    std::complex<double> norm(size_t num_dets_to_use);
    std::vector<std::complex<double>> build_one_rdm(size_t num_dets_to_use);
    std::vector<std::complex<double>> compute_variational_energy(size_t num_dets_to_use);
    energy_t slater_condon0(
        std::vector<int> &occs, std::vector<std::complex<double>> &h1e, std::vector<std::complex<double>> &h2e);
    energy_t slater_condon1(
        std::vector<int> &occs,
        Excitation &excit_ia,
        std::vector<std::complex<double>> &h1e,
        std::vector<std::complex<double>> &h2e);
    energy_t slater_condon2(Excitation &ijab, std::vector<std::complex<double>> &h2e);

   private:
    size_t flat_indx(size_t p, size_t q);
    size_t flat_indx(size_t p, size_t q, size_t r, size_t s);
    std::pair<size_t, size_t> map_orb_to_spat_spin(size_t p);
    std::vector<std::complex<double>> coeffs;
    std::vector<BitString> dets;
    std::vector<std::complex<double>> energy_accumulator;
    size_t num_dets;
    size_t num_elec;
    size_t num_spatial;
};
}  // namespace ipie

#endif
