#include "wavefunction.h"

#include <complex>
#include <iomanip>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"
#include "hamiltonian.h"

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

ipie::complex_t Wavefunction::norm() {
    ipie::complex_t norm = 0.0;
    for (const auto &[det_ket, coeff_ket] : map) {
        norm += conj(coeff_ket) * coeff_ket;
    }
    return sqrt(norm);
}

std::vector<ipie::complex_t> Wavefunction::build_one_rdm() {
    std::vector<size_t> occs(num_elec);
    ipie::complex_t denom = 0.0;
    std::vector<ipie::complex_t> density_matrix(2 * num_spatial * num_spatial);
    for (const auto &[det_ket, coeff_ket] : map) {
        denom += conj(coeff_ket) * coeff_ket;
        det_ket.decode_bits(occs);
        for (size_t iel = 0; iel < num_elec; iel++) {
            int spatial = occs[iel] / 2;
            int spin_offset = num_spatial * num_spatial * (occs[iel] % 2);
            int pq = spatial * num_spatial + spatial + spin_offset;
            density_matrix[pq] += conj(coeff_ket) * coeff_ket;
        }
        BitString det_bra(det_ket);
        for (size_t i = 0; i < occs.size(); i++) {
            indx_t i_spat_spin = map_orb_to_spat_spin(occs[i]);
            for (size_t a = 0; a < det_bra.num_bits; a++) {
                if (!det_ket.is_set(a) && a != occs[i]) {
                    indx_t a_spat_spin = map_orb_to_spat_spin(a);
                    if (i_spat_spin.second == a_spat_spin.second) {
                        det_bra.set_bit(a);
                        det_bra.clear_bit(occs[i]);
                        Excitation excit_ia{{occs[i]}, {a}};
                        if (map.find(det_bra) != map.end()) {
                            ipie::complex_t bra_coeff = map[det_bra];
                            ipie::complex_t perm{(double)single_excitation_permutation(det_ket, excit_ia), 0.0};
                            ipie::complex_t val = 0.5 * ipie::complex_t{perm} * conj(bra_coeff) * coeff_ket;
                            int spin_offset = num_spatial * num_spatial * i_spat_spin.second;
                            int pq = a_spat_spin.first * num_spatial + i_spat_spin.first + spin_offset;
                            int qp = i_spat_spin.first * num_spatial + a_spat_spin.first + spin_offset;
                            density_matrix[pq] += val;
                            density_matrix[qp] += conj(val);
                        }
                        det_bra.set_bit(occs[i]);
                        det_bra.clear_bit(a);
                    }
                }
            }
        }
    }
    for (size_t i = 0; i < num_spatial * num_spatial * 2; i++) {
        density_matrix[i] = density_matrix[i] / denom;
    }
    return density_matrix;
}

}  // namespace ipie