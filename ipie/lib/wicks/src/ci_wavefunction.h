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
    CIWavefunction(
        std::vector<std::complex<double>> &ci_coeffs,
        std::vector<std::vector<int>> &occa,
        std::vector<std::vector<int>> &occb,
        size_t nspatial);
    CIWavefunction(std::vector<ipie::complex_t> &ci_coeffs, std::vector<BitString> &determinants);

    std::complex<double> norm(size_t num_dets_to_use);
    std::vector<std::complex<double>> build_one_rdm(size_t num_dets_to_use);
    std::vector<std::complex<double>> compute_variational_energy(size_t num_dets_to_use);
    ipie::energy_t slater_condon0(
        std::vector<int> &occs, std::vector<std::complex<double>> &h1e, std::vector<std::complex<double>> &h2e);
    ipie::energy_t slater_condon1(
        std::vector<int> &occs,
        Excitation &excit_ia,
        std::vector<std::complex<double>> &h1e,
        std::vector<std::complex<double>> &h2e);
    ipie::energy_t slater_condon2(Excitation &ijab, std::vector<std::complex<double>> &h2e);
    // comparitor
    friend bool operator==(const CIWavefunction &lhs, const CIWavefunction &rhs);
    friend std::ostream &operator<<(std::ostream &os, const CIWavefunction &wfn);

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
