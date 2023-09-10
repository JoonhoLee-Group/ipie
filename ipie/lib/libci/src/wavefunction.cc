#include "wavefunction.h"

#include <complex>
#include <iomanip>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"

namespace ipie {

Wavefunction::Wavefunction(std::unordered_map<ipie::BitString, ipie::complex_t, ipie::BitStringHasher> det_map)
    : map(std::move(det_map)) {
    if (map.size() > 0) {
        num_spatial = map.begin()->first.num_bits / 2;
        num_elec = map.begin()->first.count_set_bits();
        num_dets = map.size();
    }
}

Wavefunction Wavefunction::build_wavefunction_from_occ_list(
    std::vector<ipie::complex_t> &ci_coeffs,
    std::vector<std::vector<int>> &occa,
    std::vector<std::vector<int>> &occb,
    size_t nspatial) {
    size_t num_spatial = nspatial;
    ipie::det_map dmap;
    for (size_t i = 0; i < ci_coeffs.size(); i++) {
        BitString det_i(2 * num_spatial);
        for (size_t a = 0; a < occa[i].size(); a++) {
            det_i.set_bit(2 * occa[i][a]);
        }
        for (size_t b = 0; b < occb[i].size(); b++) {
            det_i.set_bit(2 * occb[i][b] + 1);
        }
        if (dmap.size() == 0 || dmap.find(det_i) == dmap.end()) {
            // we don't want duplicate keys.
            dmap.insert({det_i, ci_coeffs[i]});
        } else if (dmap.find(det_i) != dmap.end()) {
            std::cout << "LIBCI::WARNING:: Found duplicate determinants during wavefunction construction." << std::endl;
        }
    }
    return Wavefunction(dmap);
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
        if (map != other.map) {
            return false;
        }
    }
    return true;
}
bool Wavefunction::operator!=(const Wavefunction &other) const {
    return !(*this == other);
}

std::ostream &operator<<(std::ostream &os, const Wavefunction &wfn) {
    for (const auto &[key, val] : wfn.map) {
        os << std::fixed << std::setprecision(4) << key << " " << val << " \n";
    }
    return os;
}

}  // namespace ipie