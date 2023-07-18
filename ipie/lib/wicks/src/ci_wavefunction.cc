#ifndef _CI_WAVEFUNCTION_H
#define _CI_WAVEFUNCTION_H


#include <complex>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"
#include "ci_wavefunction.h"

namespace ipie {

CIWavefunction::CIWavefunction(std::vector<complex_t> &ci_coeffs, std::vector<BitString> &determinants)
    : coeffs(ci_coeffs), dets(determinants) {
    num_dets = ci_coeffs.size();
    num_spatial = dets[0].num_bits;
    num_elec = dets[0].count_set_bits();
}
CIWavefunction CIWavefunction::build_ci_wavefunction(
    std::vector<complex_t> &ci_coeffs, std::vector<int> &occa, std::vector<int> &occb) {
}

complex_t CIWavefunction::norm(size_t num_dets_to_use) {
    complex_t norm = 0.0;
    for (size_t idet = 0; idet < num_dets; idet++) {
        norm += conj(coeffs[idet]) * coeffs[idet];
    }
    return norm;
}

size_t CIWavefunction::flat_indx(size_t p, size_t q) {
    return p * num_spatial + q;
}

size_t CIWavefunction::flat_indx(size_t p, size_t q, size_t r, size_t s) {
    return p * num_spatial * num_spatial * num_spatial + q * num_spatial * num_spatial + r * num_spatial + s;
}

std::pair<size_t, size_t> CIWavefunction::map_orb_to_spat_spin(size_t p) {
    std::pair<size_t, size_t> spat_spin = std::make_pair(p / 2, p % 2);
    return spat_spin;
}

energy_t CIWavefunction::slater_condon0(
    std::vector<int> &occs, std::vector<complex_t> &h1e, std::vector<complex_t> &h2e) {
    energy_t hmatel;
    for (size_t p = 0; p < occs.size(); p++) {
        indx_t p_spat_spin = map_orb_to_spat_spin(p);
        size_t p_ind = flat_indx(p_spat_spin.first, p_spat_spin.first);
        std::get<0>(hmatel) += h1e[p_ind];
        std::get<1>(hmatel) += h1e[p_ind];
        for (size_t q = p + 1; q < occs.size(); q++) {
            indx_t q_spat_spin = map_orb_to_spat_spin(q);
            size_t ijij = flat_indx(p_spat_spin.first, q_spat_spin.first, p_spat_spin.first, q_spat_spin.first);
            std::get<0>(hmatel) += h2e[ijij];
            std::get<2>(hmatel) += h2e[ijij];
            if (p_spat_spin.second == q_spat_spin.second) {
                size_t ijji = flat_indx(p_spat_spin.first, q_spat_spin.first, q_spat_spin.first, p_spat_spin.first);
                std::get<0>(hmatel) -= h2e[ijji];
                std::get<2>(hmatel) -= h2e[ijji];
            }
        }
    }
    return hmatel;
}
energy_t CIWavefunction::slater_condon1(
    std::vector<int> &occs, Excitation &excit_ia, std::vector<complex_t> &h1e, std::vector<complex_t> &h2e) {
    energy_t hmatel;
    indx_t i_spat_spin = map_orb_to_spat_spin(excit_ia.from[0]);
    indx_t a_spat_spin = map_orb_to_spat_spin(excit_ia.to[0]);
    size_t ia = flat_indx(i_spat_spin.first, a_spat_spin.first);
    std::get<0>(hmatel) = h1e[ia];
    std::get<1>(hmatel) = h1e[ia];
    for (size_t j = 0; j < num_elec; j++) {
        size_t occ_j = occs[j];
        indx_t occ_j_spat_spin = map_orb_to_spat_spin(occ_j);
        if (occ_j != excit_ia.from[0]) {
            size_t ijaj = flat_indx(i_spat_spin.first, occ_j_spat_spin.first, a_spat_spin.first, occ_j_spat_spin.first);
            std::get<0>(hmatel) += h2e[ijaj];
            std::get<2>(hmatel) += h2e[ijaj];
            if (occ_j_spat_spin.second == i_spat_spin.second) {
                size_t ijja =
                    flat_indx(i_spat_spin.first, occ_j_spat_spin.first, occ_j_spat_spin.first, a_spat_spin.first);
                std::get<0>(hmatel) -= h2e[ijja];
                std::get<2>(hmatel) -= h2e[ijja];
            }
        }
    }
    return hmatel;
}
energy_t CIWavefunction::slater_condon2(Excitation &ijab, std::vector<complex_t> &h2e) {
    energy_t hmatel;
    return hmatel;
    // def slater_condon2(system, i, j, a, b, perm):
    //     ii, si = i
    //     jj, sj = j
    //     aa, sa = a
    //     bb, sb = b
    //     hmatel = 0.0
    //     if si == sa:
    //         hmatel = system.hijkl(ii, jj, aa, bb)
    //     if si == sb:
    //         hmatel -= system.hijkl(ii, jj, bb, aa)
    //     if perm:
    //         return -hmatel
    //     else:
    //         return hmatel
    // indx_t i_spat_spin = map_orb_to_spat_spin();
    // indx_t j_spat_spin = map_orb_to_spat_spin(excit_ia.first);
    // indx_t a_spat_spin = map_orb_to_spat_spin(excit_ia.second);
}

std::vector<complex_t> CIWavefunction::build_one_rdm(size_t num_dets_to_use) {
    Excitation ia(1);
    std::vector<int> occs(num_elec);
    complex_t denom = 0.0;
    std::vector<complex_t> density_matrix(2 * num_spatial * num_spatial);
    for (size_t idet = 0; idet < num_dets; idet++) {
        BitString det_ket = dets[idet];
        complex_t coeff_ket = coeffs[idet];
        denom += conj(coeff_ket) * coeff_ket;
        det_ket.decode_bits(occs);
        // for (auto i : occs) {
        //     std::cout << i << std::endl;
        // }
        for (int iel = 0; iel < num_elec; iel++) {
            int spatial = occs[iel] / 2;
            int spin_offset = num_spatial * num_spatial * (occs[iel] % 2);
            int pq = spatial * num_spatial + spatial + spin_offset;
            density_matrix[pq] += conj(coeff_ket) * coeff_ket;
            // if (abs(density_matrix[pq]) > 0) {
            //     std::cout << pq << " " << spatial << " " << density_matrix[pq] << std::endl;
            // }
        }
        for (int jdet = idet + 1; jdet < num_dets; jdet++) {
            BitString det_bra = dets[jdet];
            complex_t coeff_bra = coeffs[idet];
            int excitation = det_bra.count_difference(det_ket);
            if (excitation == 1) {
                decode_single_excitation(det_bra, det_ket, ia);
                int perm = single_excitation_permutation(det_ket, ia);
                int si = ia.from[0] % 2;
                int sa = ia.to[0] % 2;
                int spat_i = ia.from[0] / 2;
                int spat_a = ia.to[0] / 2;
                if (si == sa) {
                    int spin_offset = num_spatial * num_spatial * si;
                    int pq = spat_a * num_spatial + spat_i + spin_offset;
                    int qp = spat_i * num_spatial + spat_a + spin_offset;
                    complex_t val = (double)perm * conj(coeff_bra) * coeff_ket;
                    density_matrix[pq] += val;
                    density_matrix[qp] += conj(val);
                }
            }
        }
    }
    for (int i = 0; i < num_spatial * num_spatial * 2; i++) {
        density_matrix[i] = density_matrix[i] / denom;
    }
    return density_matrix;
}
std::vector<complex_t> CIWavefunction::compute_variational_energy(size_t num_dets_to_use) {
    std::vector<complex_t> x;
    return x;
}

}  // namespace ipie

#endif