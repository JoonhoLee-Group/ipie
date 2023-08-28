#include "wavefunction.h"

#include <complex>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"

namespace ipie {

Wavefunction::Wavefunction(
    std::vector<ipie::complex_t> &ci_coeffs,
    std::vector<std::vector<int>> &occa,
    std::vector<std::vector<int>> &occb,
    size_t nspatial)
    : coeffs(ci_coeffs) {
    num_dets = ci_coeffs.size();
    num_spatial = nspatial;
    num_elec = occa[0].size() + occb[0].size();
    dets.resize(num_dets, BitString(num_spatial));
    for (size_t i = 0; i < ci_coeffs.size(); i++) {
        BitString det_i(2 * num_spatial);
        for (size_t a = 0; a < occa[i].size(); a++) {
            det_i.set_bit(2 * a);
        }
        for (size_t b = 0; b < occb[i].size(); b++) {
            det_i.set_bit(2 * b + 1);
        }
        dets[i] = det_i;
    }
}

Wavefunction::Wavefunction(std::vector<ipie::complex_t> &ci_coeffs, std::vector<BitString> &determinants)
    : coeffs(ci_coeffs), dets(determinants) {
    num_spatial = dets[0].num_bits / 2;
    num_elec = dets[0].count_set_bits();
    num_dets = ci_coeffs.size();
}

ipie::complex_t Wavefunction::norm(size_t num_dets_to_use) {
    ipie::complex_t norm = 0.0;
    for (size_t idet = 0; idet < num_dets; idet++) {
        norm += conj(coeffs[idet]) * coeffs[idet];
    }
    return norm;
}

bool operator==(const Wavefunction &lhs, const Wavefunction &rhs) {
    if (lhs.num_dets != rhs.num_dets) {
        return false;
    } else if (lhs.num_spatial != rhs.num_spatial) {
        return false;
    } else if (lhs.num_elec != rhs.num_elec) {
        return false;
    } else {
        for (size_t idet = 0; idet < lhs.num_dets; idet++) {
            if (lhs.dets[idet] != rhs.dets[idet]) {
                return false;
            }
            if (abs(lhs.coeffs[idet] - rhs.coeffs[idet]) > 1e-12) {
                return false;
            }
        }
    }
    return true;
};
std::ostream &operator<<(std::ostream &os, const Wavefunction &wfn) {
    for (size_t idet = 0; idet < wfn.num_dets; idet++) {
        os << wfn.coeffs[idet] << " " << wfn.dets[idet] << " \n";
    }
    return os;
}

}  // namespace ipie