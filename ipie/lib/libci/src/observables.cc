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
    energy_t tmp;
    return tmp;
}

}  // namespace ipie