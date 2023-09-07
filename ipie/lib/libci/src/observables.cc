#include <complex>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"
#include "hamiltonian.h"
#include "wavefunction.h"
namespace ipie {

std::vector<ipie::complex_t> build_one_rdm(Wavefunction &wfn) {
    Excitation ia(1);
    std::vector<int> occs(wfn.num_elec);
    ipie::complex_t denom = 0.0;
    size_t num_spatial = wfn.num_spatial;
    std::vector<ipie::complex_t> density_matrix(2 * num_spatial * num_spatial);
    for (size_t idet = 0; idet < wfn.num_dets; idet++) {
        BitString det_ket = wfn.dets[idet];
        ipie::complex_t coeff_ket = wfn.coeffs[idet];
        denom += conj(coeff_ket) * coeff_ket;
        det_ket.decode_bits(occs);
        // std::cout << det_ket << std::endl;
        for (size_t iel = 0; iel < wfn.num_elec; iel++) {
            int spatial = occs[iel] / 2;
            int spin_offset = num_spatial * num_spatial * (occs[iel] % 2);
            int pq = spatial * num_spatial + spatial + spin_offset;
            density_matrix[pq] += conj(coeff_ket) * coeff_ket;
        }
        for (size_t jdet = idet + 1; jdet < wfn.num_dets; jdet++) {
            BitString det_bra = wfn.dets[jdet];
            ipie::complex_t coeff_bra = wfn.coeffs[jdet];
            int excitation = det_bra.count_difference(det_ket);
            if (excitation == 1) {
                decode_single_excitation(det_bra, det_ket, ia);
                int perm = single_excitation_permutation(det_ket, ia);
                indx_t i_spat_spin = map_orb_to_spat_spin(ia.from[0]);
                indx_t a_spat_spin = map_orb_to_spat_spin(ia.to[0]);
                if (i_spat_spin.second == a_spat_spin.second) {
                    int spin_offset = num_spatial * num_spatial * i_spat_spin.second;
                    int pq = a_spat_spin.first * num_spatial + i_spat_spin.first + spin_offset;
                    int qp = i_spat_spin.first * num_spatial + a_spat_spin.first + spin_offset;
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

energy_t compute_variational_energy(Wavefunction &wfn, Hamiltonian &ham) {
    energy_t var_eng;
    std::vector<int> occs;
    // ipie::complex_t norm;
    // for (size_t ket_indx = 0; ket_indx < wfn.num_dets; ket_indx++) {
    //     // 1. loop over connected determinants
    //     BitString det_ket = wfn.dets[ket_indx];  // copy?
    //     det_ket.decode_bits(occs);
    //     ipie::complex_t fac = conj(wfn.map[det_ket]) * wfn.coeffs.map[det_ket];
    //     var_eng += fac * slater_condon0(ham, occs);
    //     norm += fac;
    //     BitString det_bra(det_ket);
    //     for (size_t i = 0; i < occs.size(); i++) {
    //         for (size_t a = 0; a < det_bra.num_bits; a++) {
    //             if (det_bra.is_set(i) && !det_bra.is_set(a)) {
    //                 det_bra.clear_bit(i);
    //                 det_bra.set_bit(a);
    //                 Excitation excit_ia{{i}, {a}};
    //                 ipie::complex_t bra_coeff = wfn.map(det_bra);
    //                 int perm = single_excitation_permutation(det_ket, excit_ia);
    //                 var_eng += perm * conj(bra_coeff) * wfn.coeffs[ket_indx] * slater_condon1(ham, occs, excit_ia);
    //                 // reset to det_ket
    //                 det_bra.clear_bit(a);
    //                 det_bra.set_bit(i);
    //             }
    //         }
    //         // TODO: optimize for symmetry
    //         // for (size_t j = 0; j < occs.size(); j++) {
    //         //     std::vector<indx_t> doubles = ham.get_doubles(i, j);
    //         //     for (auto ab : doubles) {
    //         //         det_bra.clear_bit(i);
    //         //         det_bra.clear_bit(j);
    //         //         det_bra.set_bit(std::get<0>(ab));
    //         //         det_bra.set_bit(std::get<1>(ab));
    //         //         Excitation excit_ijab({i, j}, {a, b});
    //         //         size_t bra_indx = wfn.det_indx(det_bra);
    //         //         int perm = double_excitation_permutation(det_ket, excit_ijab);
    //         //         hmatel +=
    //         //             perm * conj(wfn.coeffs[bra_indx]) * wfn.coeffs[ket_indx] * slater_condon2(ham, occs,
    //         //             excit_ia);
    //         //         // reset to det_ket
    //         //         det_bra.set_bit(i);
    //         //         det_bra.set_bit(j);
    //         //         det_bra.clear_bit(std::get<0>(ab));
    //         //         det_bra.clear_bit(std::get<1>(ab));
    //         //     }
    //         // }
    //     }
    // }
    return var_eng;
}

}  // namespace ipie