#include "wavefunction.h"

#include <complex>
#include <iomanip>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "determinant.h"
#include "excitations.h"
#include "hamiltonian.h"

namespace ipie {

Wavefunction::Wavefunction(
    ipie::det_map detr_map,
    ipie::bs_map unique_det_a,
    ipie::bs_map unique_det_b,
    ipie::bs_map epq_alpha,
    ipie::bs_map epq_beta)
    : dmap(std::move(detr_map)),
      map_a(std::move(unique_det_a)),
      map_b(std::move(unique_det_b)),
      epq_a(std::move(epq_alpha)),
      epq_b(std::move(epq_beta)) {
    if (dmap.size() > 0) {
        num_spatial = dmap.begin()->first.alpha.num_bits;
        num_alpha = dmap.begin()->first.alpha.count_set_bits();
        num_beta = dmap.begin()->first.beta.count_set_bits();
        num_elec = num_alpha + num_beta;
        num_dets = dmap.size();
    }
}

Wavefunction Wavefunction::build_wavefunction_from_occ_list(
    std::vector<ipie::complex_t> &ci_coeffs,
    std::vector<std::vector<size_t>> &occa,
    std::vector<std::vector<size_t>> &occb,
    size_t nspatial) {
    size_t num_spatial = nspatial;
    ipie::bs_map dmap_a, dmap_b;
    std::vector<ipie::Determinant> dets;
    ipie::det_map dmap;
    // dmap_a dmap_b
    // dmap_a[alpha_bs] = {c_0, c_1, .., c_l} : ci coefficients of determinants with common alpha_bs
    for (size_t i = 0; i < ci_coeffs.size(); i++) {
        BitString det_a(num_spatial);
        BitString det_b(num_spatial);
        for (size_t a = 0; a < occa[i].size(); a++) {
            det_a.set_bit(occa[i][a]);
        }
        for (size_t b = 0; b < occb[i].size(); b++) {
            det_b.set_bit(occb[i][b]);
        }
        auto det = Determinant(det_a, det_b);
        dets.push_back(det);
        dmap_a[det_a].push_back(det);
        dmap_b[det_b].push_back(det);
        dmap[det] = ci_coeffs[i];
    }
    // build single excitation links from the unique alpha strings
    ipie::bs_map epq_alpha;
    for (const auto &[key_i, value_i] : dmap_a) {
        for (const auto &[key_j, value_j] : dmap_a) {
            if (key_i.count_difference(key_j) == 1) {
                for (auto i_con_beta : value_j) {
                    epq_alpha[key_i].push_back(i_con_beta);
                }
            }
        }
    }
    // build single excitation links from the unique beta strings
    ipie::bs_map epq_beta;
    for (const auto &[key_i, value_i] : dmap_b) {
        for (const auto &[key_j, value_j] : dmap_b) {
            if (key_i.count_difference(key_j) == 1) {
                for (auto i_con_alpha : value_j) {
                    epq_beta[key_i].push_back(i_con_alpha);
                }
            }
        }
    }
    ipie::Wavefunction wfn(dmap, dmap_a, dmap_b, epq_alpha, epq_beta);
    return wfn;
}

bool Wavefunction::operator==(const Wavefunction &other) const {
    if (num_dets != other.num_dets) {
        return false;
    } else if (num_spatial != other.num_spatial) {
        return false;
    } else if (num_elec != other.num_elec) {
        return false;
    } else {
        if (map_a != other.map_a) {
            return false;
        }
        if (map_b != other.map_b) {
            return false;
        }
    }
    return true;
}
bool Wavefunction::operator!=(const Wavefunction &other) const {
    return !(*this == other);
}

std::ostream &operator<<(std::ostream &os, const Wavefunction &wfn) {
    for (const auto &[det, coeff] : wfn.dmap) {
        os << std::fixed << std::setprecision(4) << det.alpha << " " << det.beta << " " << coeff << " \n";
    }
    return os;
}

ipie::complex_t Wavefunction::norm() {
    ipie::complex_t norm = 0.0;
    for (const auto &[det, coeff] : dmap) {
        norm += conj(coeff) * coeff;
    }
    return sqrt(norm);
}

std::vector<ipie::complex_t> Wavefunction::build_one_rdm() {
    std::vector<ipie::complex_t> density_matrix(2 * num_spatial * num_spatial);
    std::vector<size_t> occs_a(num_alpha);
    ipie::complex_t denom = 0.0;
    ipie::Excitation ia(1);
    for (const auto &[det_ket_a, unique_dets] : map_a) {
        det_ket_a.decode_bits(occs_a);
        // diagonal part
        for (auto det_ket : unique_dets) {
            ipie::complex_t coeff_ket = dmap[det_ket];
            ipie::complex_t c_sq = conj(coeff_ket) * coeff_ket;
            denom += c_sq;
            for (size_t iel = 0; iel < num_alpha; iel++) {
                int spatial = occs_a[iel];
                int pq = spatial * num_spatial + spatial;
                density_matrix[pq] += c_sq;
            }
            // i = occ, a = vert  <bra| a^i |ket> = D_{ai} (a = row, i = column)
            for (auto bra_det : epq_a[det_ket_a]) {
                if (bra_det.beta.count_difference(det_ket.beta) == 0) {
                    decode_single_excitation(bra_det.alpha, det_ket_a, ia);
                    int perm = single_excitation_permutation(det_ket_a, ia);
                    int pq = ia.to[0] * num_spatial + ia.from[0];
                    density_matrix[pq] += ipie::complex_t{(double)perm} * conj(dmap[bra_det]) * coeff_ket;
                }
            }
        }
    }
    std::vector<size_t> occs_b(num_beta);
    size_t offset = num_spatial * num_spatial;
    for (const auto &[det_ket_b, unique_dets] : map_b) {
        det_ket_b.decode_bits(occs_b);
        // diagonal part
        for (auto det_ket : unique_dets) {
            ipie::complex_t coeff_ket = dmap[det_ket];
            ipie::complex_t c_sq = conj(coeff_ket) * coeff_ket;
            for (size_t iel = 0; iel < num_beta; iel++) {
                int spatial = occs_b[iel];
                int pq = spatial * num_spatial + spatial + offset;
                density_matrix[pq] += c_sq;
            }
            // i = occ, a = vert  <bra| a^i |ket> = D_{ai} (a = row, i = column)
            for (auto bra_det : epq_b[det_ket_b]) {
                if (bra_det.alpha.count_difference(det_ket.alpha) == 0) {
                    decode_single_excitation(bra_det.beta, det_ket_b, ia);
                    int perm = single_excitation_permutation(det_ket_b, ia);
                    int pq = ia.to[0] * num_spatial + ia.from[0] + offset;
                    density_matrix[pq] += ipie::complex_t{(double)perm} * conj(dmap[bra_det]) * coeff_ket;
                }
            }
        }
    }
    for (size_t i = 0; i < 2 * num_spatial * num_spatial; i++) {
        density_matrix[i] = density_matrix[i] / denom;
    }
    return density_matrix;
}

ipie::energy_t contract_sigma_same_spin(
    size_t num_elec_spin,
    size_t num_spatial,
    size_t spin,
    ipie::bs_map &map,
    ipie::det_map &dmap,
    ipie::Hamiltonian &ham) {
    std::vector<size_t> occs(num_elec_spin);
    ipie::Excitation rs(1), pq(1);
    int perm_pq, perm_rs;
    ipie::energy_t var_eng{0, 0, 0};
    // Something is very broken here.
    for (const auto &[det_ket, unique_dets] : map) {
        det_ket.decode_bits(occs);
        // Loop over |I>
        for (auto unq_det : unique_dets) {
            ipie::Determinant det_rs = unq_det;
            for (size_t iel = 0; iel < num_elec_spin; iel++) {
                // Epq |I> (q in occ_I)
                int s = occs[iel];
                det_rs.clear_bit(s, spin);
                for (size_t r = 0; r < num_spatial; r++) {
                    det_rs.set_bit(r, spin);
                    auto ket_rs_it = dmap.find(det_rs);
                    if (ket_rs_it != dmap.end()) {
                        rs.from[0] = s;
                        rs.to[0] = r;
                        perm_rs = single_excitation_permutation(det_ket, rs);
                        ipie::complex_t fac =
                            ipie::complex_t{(double)perm_rs} * conj(ket_rs_it->second) * dmap.find(unq_det)->second;
                        var_eng += ipie::energy_t{
                            0,
                            fac * ham.get_h1e(rs.to[0], rs.from[0]),
                            -0.5 * fac * ham.get_h1e_c(rs.to[0], rs.from[0])};
                    }  // End Ers|I>
                    // Epq Ers|I>
                    // TODO decode Ers|I>
                    if (det_rs.count_set_bits(spin) != num_elec_spin)
                        continue;
                    for (size_t q = 0; q < num_spatial; q++) {
                        ipie::Determinant det_pqrs = det_rs;
                        det_pqrs.clear_bit(q, spin);
                        for (size_t p = 0; p < num_spatial; p++) {
                            if (det_pqrs.is_set(p, spin)) {
                                continue;
                            }
                            det_pqrs.set_bit(p, spin);
                            auto ket_pqrs_it = dmap.find(det_pqrs);
                            if (ket_pqrs_it != dmap.end()) {
                                pq.from[0] = q;
                                pq.to[0] = p;
                                rs.from[0] = s;
                                rs.to[0] = r;
                                perm_rs = single_excitation_permutation(det_ket, rs);
                                perm_pq = single_excitation_permutation(det_rs[spin], pq);
                                ipie::complex_t fac = ipie::complex_t{(double)perm_rs * perm_pq} *
                                                      conj(ket_pqrs_it->second) * dmap.find(unq_det)->second;
                                var_eng += ipie::energy_t{
                                    0, 0, 0.5 * fac * ham.get_h2e(pq.to[0], pq.from[0], rs.to[0], rs.from[0])};
                            }
                            det_pqrs.clear_bit(p, spin);
                        }  // loop over p
                        det_pqrs.set_bit(q, spin);
                    }  // loop over q
                    det_rs.clear_bit(r, spin);
                }  // loop over r
                det_rs.set_bit(s, spin);
            }  // loop over s
        }
    }
    return var_eng;
}

ipie::energy_t contract_sigma_opp_spin(
    size_t spin_a,
    size_t spin_b,
    ipie::bs_map &map_a,
    ipie::bs_map &map_b,
    ipie::det_map &dmap,
    ipie::Hamiltonian &ham) {
    size_t num_a = map_a.begin()->first.count_set_bits();
    size_t num_b = map_b.begin()->first.count_set_bits();
    size_t num_spatial = map_a.begin()->first.num_bits;
    std::vector<size_t> occ_a(num_a), occ_b(num_b);
    ipie::Excitation rs(1), pq(1);
    int perm_pq, perm_rs;
    ipie::energy_t var_eng{0, 0, 0};
    for (const auto &[det_ket, unique_dets] : map_a) {
        det_ket.decode_bits(occ_a);
        // Loop over |I_a>
        for (size_t iel = 0; iel < num_a; iel++) {
            // Epq |I> (q in occ_I)
            int s = occ_a[iel];
            for (size_t r = 0; r < num_spatial; r++) {
                for (auto unq_det : unique_dets) {
                    ipie::Determinant det_rs = unq_det;
                    det_rs.clear_bit(s, spin_a);
                    det_rs.set_bit(r, spin_a);
                    // Epq^b Ers^a|I>
                    // TODO decode Ers|I>
                    if (det_rs.count_set_bits(spin_a) != num_a)
                        continue;
                    for (size_t q = 0; q < num_spatial; q++) {
                        ipie::Determinant det_pqrs = det_rs;
                        det_pqrs.clear_bit(q, spin_b);
                        for (size_t p = 0; p < num_spatial; p++) {
                            if (det_pqrs.is_set(p, spin_b)) {
                                continue;
                            }
                            det_pqrs.set_bit(p, spin_b);
                            auto ket_pqrs_it = dmap.find(det_pqrs);
                            if (ket_pqrs_it != dmap.end()) {
                                pq.from[0] = q;
                                pq.to[0] = p;
                                rs.from[0] = s;
                                rs.to[0] = r;
                                perm_rs = single_excitation_permutation(det_ket, rs);
                                perm_pq = single_excitation_permutation(det_rs[spin_b], pq);
                                ipie::complex_t fac = ipie::complex_t{(double)perm_rs * perm_pq} *
                                                      conj(ket_pqrs_it->second) * dmap.find(unq_det)->second;
                                var_eng += ipie::energy_t{
                                    0, 0, 0.5 * fac * ham.get_h2e(pq.to[0], pq.from[0], rs.to[0], rs.from[0])};
                            }  // End Epq Ers|I>
                            det_pqrs.clear_bit(p, spin_b);
                        }
                        det_pqrs.set_bit(q, spin_b);
                    }
                    det_rs.clear_bit(r, spin_a);
                    det_rs.set_bit(s, spin_a);
                }
            }
        }
    }
    return var_eng;
}

ipie::energy_t Wavefunction::energy(Hamiltonian &ham) {
    energy_t var_eng{0.0, 0.0, 0.0};
    ipie::complex_t denom = norm();
    denom *= denom;
    var_eng += contract_sigma_same_spin(num_alpha, num_spatial, 0, map_a, dmap, ham);
    var_eng += contract_sigma_same_spin(num_beta, num_spatial, 1, map_b, dmap, ham);
    var_eng += contract_sigma_opp_spin(0, 1, map_a, map_b, dmap, ham);
    var_eng += contract_sigma_opp_spin(1, 0, map_b, map_a, dmap, ham);
    var_eng /= denom;
    var_eng.e1b += ham.e0;
    var_eng.etot = var_eng.e1b + var_eng.e2b;
    return var_eng;
}

}  // namespace ipie