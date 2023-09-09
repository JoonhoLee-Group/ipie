#include "wavefunction.h"

#include <complex>
#include <iomanip>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"

namespace ipie {

Wavefunction::Wavefunction(std::vector<ipie::complex_t> ci_coeffs, std::vector<BitString> determinants)
    : coeffs(ci_coeffs), dets(determinants) {
    num_spatial = dets[0].num_bits / 2;
    num_elec = dets[0].count_set_bits();
    num_dets = ci_coeffs.size();
}
Wavefunction::Wavefunction(std::unordered_map<ipie::BitString, ipie::complex_t, ipie::BitStringHasher> det_map)
    : map(std::move(det_map)) {
    num_spatial = map.begin()->first.num_bits / 2;
    num_elec = map.begin()->first.count_set_bits();
    num_dets = map.size();
}

Wavefunction Wavefunction::build_wavefunction_from_occ_list(
    std::vector<ipie::complex_t> &ci_coeffs,
    std::vector<std::vector<int>> &occa,
    std::vector<std::vector<int>> &occb,
    size_t nspatial) {
    size_t num_dets = ci_coeffs.size();
    size_t num_spatial = nspatial;
    std::vector<BitString> dets;
    dets.resize(num_dets, BitString(num_spatial));
    for (size_t i = 0; i < ci_coeffs.size(); i++) {
        BitString det_i(2 * num_spatial);
        for (size_t a = 0; a < occa[i].size(); a++) {
            det_i.set_bit(2 * occa[i][a]);
        }
        for (size_t b = 0; b < occb[i].size(); b++) {
            det_i.set_bit(2 * occb[i][b] + 1);
        }
        dets[i] = det_i;
    }
    return Wavefunction(ci_coeffs, dets);
}

ipie::complex_t Wavefunction::norm() {
    ipie::complex_t norm = 0.0;
    for (const auto &[det_ket, coeff_ket] : map) {
        norm += conj(coeff_ket) * coeff_ket;
    }
    return sqrt(norm);
}

bool Wavefunction::operator==(const Wavefunction &other) const {
    if (num_dets != other.num_dets) {
        return false;
    } else if (num_spatial != other.num_spatial) {
        return false;
    } else if (num_elec != other.num_elec) {
        return false;
    } else {
        for (size_t idet = 0; idet < num_dets; idet++) {
            if (dets[idet] != other.dets[idet]) {
                return false;
            }
            if (abs(coeffs[idet] - other.coeffs[idet]) > 1e-12) {
                return false;
            }
        }
    }
    return true;
}

std::ostream &operator<<(std::ostream &os, const Wavefunction &wfn) {
    for (const auto &[key, val] : wfn.map) {
        os << std::fixed << std::setprecision(4) << key << " " << val << " \n";
    }
    return os;
}

}  // namespace ipie