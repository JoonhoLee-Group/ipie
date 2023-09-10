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
    size_t idet = 0;
    std::vector<size_t> occs(occa[0].size() + occb[0].size());
    size_t mean_conn = 0;
    std::cout << "HERE" << std::endl;
    for (const auto &[det_ket, coeff_ket] : dmap) {
        det_ket.decode_bits(occs);
        BitString ex_det(det_ket);
        for (size_t i = 0; i < occs.size(); i++) {
            for (size_t a = 0; a < det_ket.num_bits; a++) {
                for (size_t j = i + 1; j < occs.size(); j++) {
                    for (size_t b = a + 1; b < det_ket.num_bits; b++) {
                        // if (!det_ket.is_set(a) && !det_ket.is_set(b)) {
                        //     if ((occs[i] % 2 == a % 2 && occs[j] % 2 == b % 2) ||
                        //         (occs[j] % 2 == a % 2 && occs[i] % 2 == b % 2)) {
                        //         mean_conn++;
                        //         // ex_det.clear_bit(occs[i]);
                        //         // ex_det.clear_bit(occs[j]);
                        //         // ex_det.set_bit(a);
                        //         // ex_det.set_bit(b);
                        //         // if (dmap.find(ex_det) != dmap.end()) {
                        //         //     mean_conn += 1;
                        //         // }
                        //         // ex_det.set_bit(occs[i]);
                        //         // ex_det.set_bit(occs[j]);
                        //         // ex_det.clear_bit(a);
                        //         // ex_det.clear_bit(b);
                        //     }
                        // }
                        mean_conn++;
                    }
                }
            }
        }
        idet++;
    }
    std::cout << mean_conn << std::endl;
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
                            // why factor of two?
                            ipie::complex_t val = ipie::complex_t{perm} * conj(bra_coeff) * coeff_ket;
                            int spin_offset = num_spatial * num_spatial * i_spat_spin.second;
                            int pq = a_spat_spin.first * num_spatial + i_spat_spin.first + spin_offset;
                            // int qp = i_spat_spin.first * num_spatial + a_spat_spin.first + spin_offset;
                            density_matrix[pq] += val;
                            // density_matrix[qp] += conj(val);
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

ipie::energy_t Wavefunction::energy(Hamiltonian &ham) {
    energy_t var_eng{0.0, 0.0, 0.0};
    std::vector<size_t> occs(num_elec);
    ipie::complex_t norm;
    // std::cout << *this << std::endl;
    for (const auto &[det_ket, coeff_ket] : map) {
        det_ket.decode_bits(occs);
        ipie::complex_t fac = conj(coeff_ket) * coeff_ket;
        auto sc0 = slater_condon0(ham, occs);
        sc0 *= fac;
        var_eng += sc0;
        norm += fac;
        BitString det_bra(det_ket);
        // initially |bra> = |ket>
        // followed by b^ja^i|ket>
        for (size_t i = 0; i < occs.size(); i++) {
            for (size_t a = 0; a < det_bra.num_bits; a++) {
                if (!det_ket.is_set(a)) {
                    det_bra.clear_bit(occs[i]);
                    det_bra.set_bit(a);
                    Excitation excit_ia{{occs[i]}, {a}};
                    auto bra_it = map.find(det_bra);
                    if (bra_it != map.end()) {
                        ipie::complex_t perm_ia = {(double)single_excitation_permutation(det_ket, excit_ia), 0.0};

                        fac = ipie::complex_t{perm_ia} * conj(bra_it->second) * coeff_ket;
                        energy_t sc1 = slater_condon1(ham, occs, excit_ia);
                        sc1 *= fac;
                        var_eng += sc1;
                    }
                    det_bra.set_bit(occs[i]);
                    det_bra.clear_bit(a);
                }
            }
        }
        for (size_t i = 0; i < occs.size(); i++) {
            for (size_t a = 0; a < det_bra.num_bits; a++) {
                for (size_t j = i + 1; j < occs.size(); j++) {
                    for (size_t b = a + 1; b < det_bra.num_bits; b++) {
                        // size_t ij = occs[i] * 2 * num_spatial + occs[j];
                        // size_t ab = a * 2 * num_spatial + b;
                        // if (!det_ket.is_set(b) && (occs[i] > occs[j]) && (a > b) && (ij >= ab)) {
                        if (!det_ket.is_set(a) && !det_ket.is_set(b)) {
                            det_bra.clear_bit(occs[i]);
                            det_bra.set_bit(a);
                            Excitation excit_ia = {{a}, {occs[i]}};
                            ipie::complex_t perm_ijab =
                                ipie::complex_t{(double)single_excitation_permutation(det_bra, excit_ia), 0.0};
                            det_bra.clear_bit(occs[j]);
                            det_bra.set_bit(b);
                            excit_ia = {{b}, {occs[j]}};
                            perm_ijab *= ipie::complex_t{(double)single_excitation_permutation(det_bra, excit_ia), 0.0};
                            auto bra_it = map.find(det_bra);
                            if (bra_it != map.end()) {
                                Excitation excit_ijab({occs[i], occs[j]}, {a, b});
                                fac = perm_ijab * conj(bra_it->second) * coeff_ket;
                                energy_t sc2 = slater_condon2(ham, excit_ijab);
                                sc2 *= fac;
                                // // std::cout << det_bra << std::endl;
                                // std::cout << occs[i] << " " << occs[j] << " " << a << " " << b << perm_ijab << " "
                                //           << sc2 << " " << det_ket << " " << det_bra << std::endl;
                                var_eng += sc2;
                            }
                            det_bra.set_bit(occs[j]);
                            det_bra.set_bit(occs[i]);
                            det_bra.clear_bit(b);
                            det_bra.clear_bit(a);
                        }
                    }
                }
            }
        }
    }
    var_eng /= norm;
    return var_eng;
}

}  // namespace ipie