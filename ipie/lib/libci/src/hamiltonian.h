#ifndef _HAMILTONIAN_H
#define _HAMILTONIAN_H

#include <complex>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "custom_types.h"
#include "excitations.h"

namespace ipie {

typedef std::tuple<size_t, size_t, ipie::complex_t> hijkl_t;

struct Hamiltonian {
    Hamiltonian(
        std::vector<ipie::complex_t> &h1e, std::vector<ipie::complex_t> &h2e, ipie::complex_t e0, size_t num_spat);
    size_t flat_indx(size_t p, size_t q) const;
    size_t flat_indx(size_t p, size_t q, size_t r, size_t s) const;
    std::vector<ipie::complex_t> h1e;
    std::vector<ipie::complex_t> h2e;
    ipie::complex_t e0;
    size_t num_spatial;
};

std::pair<size_t, size_t> map_orb_to_spat_spin(size_t p);
ipie::energy_t slater_condon0(const Hamiltonian &ham, const std::vector<size_t> &occs);
ipie::energy_t slater_condon1(const Hamiltonian &ham, const std::vector<size_t> &occs, const Excitation &excit_ia);
ipie::energy_t slater_condon2(const Hamiltonian &ham, const Excitation &ijab);

}  // namespace ipie

#endif