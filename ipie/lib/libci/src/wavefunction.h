#ifndef _Wavefunction_H
#define _Wavefunction_H

#include <complex>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"

namespace ipie {

typedef std::unordered_map<ipie::BitString, ipie::complex_t, ipie::BitStringHasher> det_map;

struct Wavefunction {
    // constructors
    // Wavefunction(){};
    Wavefunction(std::vector<ipie::complex_t> ci_coeffs, std::vector<BitString> dets);

    Wavefunction(std::unordered_map<ipie::BitString, ipie::complex_t, ipie::BitStringHasher> determinants);

    static Wavefunction build_wavefunction_from_occ_list(
        std::vector<std::complex<double>> &ci_coeffs,
        std::vector<std::vector<int>> &occa,
        std::vector<std::vector<int>> &occb,
        size_t nspatial);

    std::complex<double> norm();
    uint64_t operator()(const BitString &bitstring) const;
    bool operator==(const Wavefunction &rhs) const;
    friend std::ostream &operator<<(std::ostream &os, const Wavefunction &wfn);
    std::vector<std::complex<double>> coeffs;
    std::vector<BitString> dets;
    std::unordered_map<ipie::BitString, ipie::complex_t, ipie::BitStringHasher> map;
    size_t num_dets;
    size_t num_elec;
    size_t num_spatial;
};

}  // namespace ipie

#endif
