#include <complex>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"
#include "hamiltonian.h"
#include "wavefunction.h"
namespace ipie {

std::vector<ipie::complex_t> build_one_rdm(Wavefunction &wfn) {
    std::vector<size_t> occs(wfn.num_elec);
    ipie::complex_t denom = 0.0;
    size_t num_spatial = wfn.num_spatial;
    std::vector<ipie::complex_t> density_matrix(2 * num_spatial * num_spatial);
    for (const auto &[det_ket, coeff_ket] : wfn.map) {
        denom += conj(coeff_ket) * coeff_ket;
        det_ket.decode_bits(occs);
        for (size_t iel = 0; iel < wfn.num_elec; iel++) {
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
                        if (wfn.map.find(det_bra) != wfn.map.end()) {
                            ipie::complex_t bra_coeff = wfn.map[det_bra];
                            ipie::complex_t perm{(double)single_excitation_permutation(det_ket, excit_ia), 0.0};
                            ipie::complex_t val = ipie::complex_t{perm} * conj(bra_coeff) * coeff_ket;
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

energy_t compute_variational_energy(Wavefunction &wfn, Hamiltonian &ham) {
    energy_t var_eng;
    std::vector<size_t> occs(wfn.num_elec);
    ipie::complex_t norm;
    for (const auto &[det_ket, coeff_ket] : wfn.map) {
        det_ket.decode_bits(occs);
        ipie::complex_t fac = conj(coeff_ket) * coeff_ket;
        auto sc0 = slater_condon0(ham, occs);
        sc0 *= fac;
        var_eng += sc0;
        norm += fac;
        BitString det_bra(det_ket);
        // initially |bra> = |ket>
        // build a^i|ket>
        // followed by b^ja^i|ket>
        for (size_t i = 0; i < occs.size(); i++) {
            for (size_t a = 0; a < det_bra.num_bits; a++) {
                if (!det_ket.is_set(a)) {
                    det_bra.clear_bit(occs[i]);
                    det_bra.set_bit(a);
                    Excitation excit_ia{{occs[i]}, {a}};
                    auto bra_it = wfn.map.find(det_bra);
                    if (bra_it != wfn.map.end()) {
                        ipie::complex_t perm{(double)single_excitation_permutation(det_ket, excit_ia), 0.0};
                        fac = ipie::complex_t{perm} * conj(bra_it->second) * coeff_ket;
                        energy_t sc1 = slater_condon1(ham, occs, excit_ia);
                        sc1 *= fac;
                        var_eng += sc1;
                    }
                    for (size_t j = 0; j < occs.size(); j++) {
                        for (size_t b = 0; b < det_bra.num_bits; b++) {
                            if (!det_ket.is_set(b)) {
                                det_bra.clear_bit(occs[j]);
                                det_bra.set_bit(b);
                                bra_it = wfn.map.find(det_bra);
                                if (bra_it != wfn.map.end()) {
                                    Excitation excit_ijab({occs[i], occs[j]}, {a, b});
                                    ipie::complex_t perm{
                                        (double)double_excitation_permutation(det_ket, excit_ijab), 0.0};
                                    fac = perm * conj(bra_it->second) * coeff_ket;
                                    energy_t sc2 = slater_condon2(ham, excit_ijab);
                                    sc2 *= fac;
                                    var_eng += sc2;
                                }
                                det_bra.set_bit(occs[j]);
                                det_bra.clear_bit(b);
                            }
                        }
                    }
                    det_bra.clear_bit(a);
                    det_bra.set_bit(occs[i]);
                }
            }
        }
        // std::cout << std::endl;
    }
    var_eng /= norm;
    return var_eng;
}

}  // namespace ipie