#ifndef _Wavefunction_H
#define _Wavefunction_H

#include <complex>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"

namespace ipie {

struct Wavefunction {
    // constructors
    Wavefunction(
        std::vector<std::complex<double>> &ci_coeffs,
        std::vector<std::vector<int>> &occa,
        std::vector<std::vector<int>> &occb,
        size_t nspatial);
    Wavefunction(std::vector<ipie::complex_t> &ci_coeffs, std::vector<BitString> &determinants);

    std::complex<double> norm(size_t num_dets_to_use);
    friend bool operator==(const Wavefunction &lhs, const Wavefunction &rhs);
    friend std::ostream &operator<<(std::ostream &os, const Wavefunction &wfn);
    std::vector<std::complex<double>> coeffs;
    std::vector<BitString> dets;
    size_t num_dets;
    size_t num_elec;
    size_t num_spatial;
};

}  // namespace ipie

#endif
