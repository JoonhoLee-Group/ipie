#ifndef _Wavefunction_H
#define _Wavefunction_H

#include <complex>
#include <unordered_map>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "excitations.h"
#include "hamiltonian.h"

namespace ipie {

typedef std::unordered_map<ipie::BitString, ipie::complex_t, ipie::BitStringHasher> det_map;
typedef std::unordered_map<ipie::BitString, std::pair<ipie::BitString, size_t>, ipie::BitStringHasher> det_map_spin;

struct Wavefunction {
    // constructors
    Wavefunction(std::unordered_map<ipie::BitString, ipie::complex_t, ipie::BitStringHasher> determinants);

    static Wavefunction build_wavefunction_from_occ_list(
        std::vector<std::complex<double>> &ci_coeffs,
        std::vector<std::vector<int>> &occa,
        std::vector<std::vector<int>> &occb,
        size_t nspatial);

    // wavefunction norm
    std::complex<double> norm();
    // one particle reduced density matrix
    std::vector<ipie::complex_t> build_one_rdm();
    ipie::energy_t energy(Hamiltonian &ham);

    uint64_t operator()(const BitString &bitstring) const;
    bool operator==(const Wavefunction &rhs) const;
    bool operator!=(const Wavefunction &rhs) const;
    friend std::ostream &operator<<(std::ostream &os, const Wavefunction &wfn);
    std::vector<std::complex<double>> coeffs;
    std::vector<BitString> dets;
    std::unordered_map<ipie::BitString, ipie::complex_t, ipie::BitStringHasher> map;
    size_t num_dets;
    size_t num_elec;
    size_t num_spatial;
};

}  // namespace ipie

#endif
