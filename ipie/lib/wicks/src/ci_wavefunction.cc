#ifndef _CI_WAVEFUNCTION_H
#define _CI_WAVEFUNCTION_H

#include "ci_wavefunction.h"

#include <complex>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"

namespace ipie {

CIWavefunction::CIWavefunction(
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

CIWavefunction::CIWavefunction(std::vector<ipie::complex_t> &ci_coeffs, std::vector<BitString> &determinants)
    : coeffs(ci_coeffs), dets(determinants) {
    num_spatial = dets[0].num_bits / 2;
    num_elec = dets[0].count_set_bits();
    num_dets = ci_coeffs.size();
}

ipie::complex_t CIWavefunction::norm(size_t num_dets_to_use) {
    ipie::complex_t norm = 0.0;
    for (size_t idet = 0; idet < num_dets; idet++) {
        norm += conj(coeffs[idet]) * coeffs[idet];
    }
    return norm;
}

bool operator==(const CIWavefunction &lhs, const CIWavefunction &rhs) {
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

ipie::energy_t CIWavefunction::slater_condon0(
    std::vector<int> &occs, std::vector<ipie::complex_t> &h1e, std::vector<ipie::complex_t> &h2e) {
    ipie::energy_t hmatel;
    for (size_t p = 0; p < occs.size(); p++) {
        indx_t p_spat_spin = map_orb_to_spat_spin(occs[p]);
        size_t p_ind = flat_indx(p_spat_spin.first, p_spat_spin.first);
        std::get<0>(hmatel) += h1e[p_ind];
        std::get<1>(hmatel) += h1e[p_ind];
        for (size_t q = p + 1; q < occs.size(); q++) {
            indx_t q_spat_spin = map_orb_to_spat_spin(occs[q]);
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

ipie::energy_t CIWavefunction::slater_condon1(
    std::vector<int> &occs,
    Excitation &excit_ia,
    std::vector<ipie::complex_t> &h1e,
    std::vector<ipie::complex_t> &h2e) {
    ipie::energy_t hmatel;
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
ipie::energy_t CIWavefunction::slater_condon2(Excitation &ijab, std::vector<ipie::complex_t> &h2e) {
    ipie::energy_t hmatel;
    indx_t i_spat_spin = map_orb_to_spat_spin(ijab.from[0]);
    indx_t j_spat_spin = map_orb_to_spat_spin(ijab.from[1]);
    indx_t a_spat_spin = map_orb_to_spat_spin(ijab.to[0]);
    indx_t b_spat_spin = map_orb_to_spat_spin(ijab.to[1]);
    if (i_spat_spin.second == a_spat_spin.second) {
        size_t ijab = flat_indx(i_spat_spin.first, j_spat_spin.first, a_spat_spin.first, b_spat_spin.first);
        std::get<2>(hmatel) = h2e[ijab];
    }
    if (i_spat_spin.second == b_spat_spin.second) {
        size_t ijba = flat_indx(i_spat_spin.first, j_spat_spin.first, b_spat_spin.first, a_spat_spin.first);
        std::get<2>(hmatel) -= h2e[ijba];
    }
    std::get<0>(hmatel) = std::get<2>(hmatel);
    return hmatel;
}

std::vector<ipie::complex_t> CIWavefunction::build_one_rdm(size_t num_dets_to_use) {
    Excitation ia(1);
    std::vector<int> occs(num_elec);
    ipie::complex_t denom = 0.0;
    std::vector<ipie::complex_t> density_matrix(2 * num_spatial * num_spatial);
    for (size_t idet = 0; idet < num_dets; idet++) {
        BitString det_ket = dets[idet];
        ipie::complex_t coeff_ket = coeffs[idet];
        denom += conj(coeff_ket) * coeff_ket;
        det_ket.decode_bits(occs);
        for (size_t iel = 0; iel < num_elec; iel++) {
            int spatial = occs[iel] / 2;
            int spin_offset = num_spatial * num_spatial * (occs[iel] % 2);
            int pq = spatial * num_spatial + spatial + spin_offset;
            density_matrix[pq] += conj(coeff_ket) * coeff_ket;
        }
        for (size_t jdet = idet + 1; jdet < num_dets; jdet++) {
            BitString det_bra = dets[jdet];
            ipie::complex_t coeff_bra = coeffs[idet];
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
                    ipie::complex_t val = (double)perm * conj(coeff_bra) * coeff_ket;
                    density_matrix[pq] += val;
                    density_matrix[qp] += conj(val);
                }
            }
        }
    }
    for (size_t i = 0; i < num_spatial * num_spatial * 2; i++) {
        density_matrix[i] = density_matrix[i] / denom;
    }
    return density_matrix;
}
std::vector<ipie::complex_t> CIWavefunction::compute_variational_energy(size_t num_dets_to_use) {
    std::vector<ipie::complex_t> x;
    return x;
}

std::ostream &operator<<(std::ostream &os, const CIWavefunction &wfn) {
    for (size_t idet = 0; idet < wfn.num_dets; idet++) {
        os << wfn.coeffs[idet] << " " << wfn.dets[idet] << " \n";
    }
    return os;
}

}  // namespace ipie

#endif