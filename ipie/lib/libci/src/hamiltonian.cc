#include "hamiltonian.h"

#include <complex>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "custom_types.h"
#include "excitations.h"

namespace ipie {

Hamiltonian::Hamiltonian(
    std::vector<ipie::complex_t> &h1e, std::vector<ipie::complex_t> &h2e, ipie::complex_t e0, size_t num_spat)
    : h1e(h1e), h2e(h2e), e0(e0), num_spatial(num_spat) {
}

size_t Hamiltonian::flat_indx(size_t p, size_t q) const {
    return p * num_spatial + q;
}

size_t Hamiltonian::flat_indx(size_t p, size_t q, size_t r, size_t s) const {
    return p * num_spatial * num_spatial * num_spatial + q * num_spatial * num_spatial + r * num_spatial + s;
}

size_t flat_indx(size_t p, size_t q, size_t r, size_t s, size_t num_spatial) {
    return p * num_spatial * num_spatial * num_spatial + q * num_spatial * num_spatial + r * num_spatial + s;
}

std::pair<size_t, size_t> map_orb_to_spat_spin(size_t p) {
    std::pair<size_t, size_t> spat_spin = std::make_pair(p / 2, p % 2);
    return spat_spin;
}

ipie::energy_t slater_condon0(const Hamiltonian &ham, const std::vector<int> &occs) {
    ipie::energy_t hmatel{0, ham.e0, 0};
    for (size_t p = 0; p < occs.size(); p++) {
        indx_t p_spat_spin = map_orb_to_spat_spin(occs[p]);
        size_t p_ind = ham.flat_indx(p_spat_spin.first, p_spat_spin.first);
        hmatel.e1b += ham.h1e[p_ind];
        for (size_t q = p + 1; q < occs.size(); q++) {
            indx_t q_spat_spin = map_orb_to_spat_spin(occs[q]);
            size_t ppqq = ham.flat_indx(p_spat_spin.first, p_spat_spin.first, q_spat_spin.first, q_spat_spin.first);
            hmatel.e2b += ham.h2e[ppqq];
            if (p_spat_spin.second == q_spat_spin.second) {
                size_t pqqp = ham.flat_indx(p_spat_spin.first, q_spat_spin.first, q_spat_spin.first, p_spat_spin.first);
                hmatel.e2b -= ham.h2e[pqqp];
            }
        }
    }
    hmatel.etot = hmatel.e1b + hmatel.e2b;
    return hmatel;
}

ipie::energy_t slater_condon1(const Hamiltonian &ham, const std::vector<int> &occs, const Excitation &excit_ia) {
    ipie::energy_t hmatel;
    indx_t i_spat_spin = map_orb_to_spat_spin(excit_ia.from[0]);
    indx_t a_spat_spin = map_orb_to_spat_spin(excit_ia.to[0]);
    size_t ia = ham.flat_indx(i_spat_spin.first, a_spat_spin.first);
    hmatel.e1b = ham.h1e[ia];
    for (size_t j = 0; j < occs.size(); j++) {
        size_t occ_j = occs[j];
        indx_t occ_j_spat_spin = map_orb_to_spat_spin(occ_j);
        if (occ_j != excit_ia.from[0]) {
            size_t iajj =
                ham.flat_indx(i_spat_spin.first, a_spat_spin.first, occ_j_spat_spin.first, occ_j_spat_spin.first);
            hmatel.e2b += ham.h2e[iajj];
            if (occ_j_spat_spin.second == i_spat_spin.second) {
                size_t ijja =
                    ham.flat_indx(i_spat_spin.first, occ_j_spat_spin.first, occ_j_spat_spin.first, a_spat_spin.first);
                hmatel.e2b -= ham.h2e[ijja];
            }
        }
    }
    hmatel.etot = hmatel.e1b + hmatel.e2b;
    return hmatel;
}

ipie::energy_t slater_condon2(const Hamiltonian &ham, const Excitation &ijab) {
    ipie::energy_t hmatel;
    indx_t i_spat_spin = map_orb_to_spat_spin(ijab.from[0]);
    indx_t j_spat_spin = map_orb_to_spat_spin(ijab.from[1]);
    indx_t a_spat_spin = map_orb_to_spat_spin(ijab.to[0]);
    indx_t b_spat_spin = map_orb_to_spat_spin(ijab.to[1]);
    if (i_spat_spin.second == a_spat_spin.second) {
        size_t iajb = ham.flat_indx(i_spat_spin.first, a_spat_spin.first, j_spat_spin.first, b_spat_spin.first);
        hmatel.e2b = ham.h2e[iajb];
    }
    if (i_spat_spin.second == b_spat_spin.second) {
        size_t ibja = ham.flat_indx(i_spat_spin.first, b_spat_spin.first, j_spat_spin.first, a_spat_spin.first);
        hmatel.e2b -= ham.h2e[ibja];
    }
    hmatel.etot = hmatel.e1b + hmatel.e2b;
    return hmatel;
}

}  // namespace ipie