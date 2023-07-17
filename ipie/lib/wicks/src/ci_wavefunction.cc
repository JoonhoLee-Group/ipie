#ifndef _CI_WAVEFUNCTION_H
#define _CI_WAVEFUNCTION_H

#include "ci_wavefunction.h"

#include <complex>
#include <vector>

#include "bitstring.h"
#include "excitations.h"

namespace ipie {

CIWavefunction::CIWavefunction(std::vector<std::complex<double>> &ci_coeffs, std::vector<BitString> &determinants)
    : coeffs(ci_coeffs), dets(determinants) {
    num_dets = ci_coeffs.size();
}
CIWavefunction CIWavefunction::build_ci_wavefunction(
    std::vector<std::complex<double>> &ci_coeffs, std::vector<int> &occa, std::vector<int> &occb) {
}

std::vector<std::complex<double>> CIWavefunction::build_one_rdm(size_t num_dets_to_use) {
    std::vector<int> ia(2);
    size_t nel = dets[0].count_set_bits();
    size_t num_orbs = dets[0].num_bits / 2;
    std::vector<int> occs(nel);
    std::complex<double> denom = 0.0;
    std::vector<std::complex<double>> density_matrix(2 * num_orbs * num_orbs);
    for (size_t idet = 0; idet < num_dets; idet++) {
        BitString det_ket = dets[idet];
        std::complex<double> coeff_ket = coeffs[idet];
        denom += conj(coeff_ket) * coeff_ket;
        det_ket.decode_bits(occs);
        // for (auto i : occs) {
        //     std::cout << i << std::endl;
        // }
        for (int iel = 0; iel < nel; iel++) {
            int spatial = occs[iel] / 2;
            int spin_offset = num_orbs * num_orbs * (occs[iel] % 2);
            int pq = spatial * num_orbs + spatial + spin_offset;
            density_matrix[pq] += conj(coeff_ket) * coeff_ket;
            // if (abs(density_matrix[pq]) > 0) {
            //     std::cout << pq << " " << spatial << " " << density_matrix[pq] << std::endl;
            // }
        }
        for (int jdet = idet + 1; jdet < num_dets; jdet++) {
            BitString det_bra = dets[jdet];
            std::complex<double> coeff_bra = coeffs[idet];
            int excitation = det_bra.count_difference(det_ket);
            if (excitation == 1) {
                decode_single_excitation(det_bra, det_ket, ia);
                int perm = single_excitation_permutation(det_ket, ia);
                int si = ia[0] % 2;
                int sa = ia[1] % 2;
                int spat_i = ia[0] / 2;
                int spat_a = ia[1] / 2;
                if (si == sa) {
                    int spin_offset = num_orbs * num_orbs * si;
                    int pq = spat_a * num_orbs + spat_i + spin_offset;
                    int qp = spat_i * num_orbs + spat_a + spin_offset;
                    std::complex<double> val = (double)perm * conj(coeff_bra) * coeff_ket;
                    density_matrix[pq] += val;
                    density_matrix[qp] += conj(val);
                }
            }
        }
    }
    for (int i = 0; i < num_orbs * num_orbs * 2; i++) {
        density_matrix[i] = density_matrix[i] / denom;
    }
    // std::cout << density_matrix[14* 14] << std::endl;
    return density_matrix;
}
std::vector<std::complex<double>> CIWavefunction::compute_variational_energy(size_t num_dets_to_use) {
    std::vector<std::complex<double>> x;
    return x;
}

}  // namespace ipie

#endif