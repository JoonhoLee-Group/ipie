#ifndef _CI_WAVEFUNCTION_H
#define _CI_WAVEFUNCTION_H

#include <vector>

#include "bitstring"

namespace ipie {

void fill_diagonal(
    BitString &det,
    double ci_coeff,
    std::vector<int> &occs,
    std::vector<std::complex<double>> &density_matrix,
    size_t num_orbs,
    size_t nel) {
    det.decode_bits(occs);
    (int iel = 0; iel < nel; iel++) {
        int spatial = occs[iel] / 2;
        int spin_offset = num_orbs * num_orbs * (occs[iel] % 2);
        int pq = spatial * num_orbs + spatial + spin_offset;
        density_matrix[pq] += ci_coeff.conj() * ci_coeff;
    }
}

CIWavefunction::CIWavefunction(std::vector<std::complex<double>> &ci_coeffs, std::vector<BitString> &dets)
    : coeffs(ci_coeffs), dets(dets) {
}
CIWavefunction CIWavefunction::build_ci_wavefunction(
    std::vector<std::complex<double>> &ci_coeffs, std::vector<int> &occa, std::vector<int> &occb) {
}

std::vector<std::complex<double>> build_one_rdm(size_t num_dets_to_use) {
}
std::vector<std::complex<double>> compute_variational_energy(size_t num_dets_to_use) {
}

}  // namespace ipie

#endif