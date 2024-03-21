#ifndef _Wavefunction_H
#define _Wavefunction_H

#include <complex>
#include <unordered_map>
#include <vector>

#include "bitstring.h"
#include "config.h"
#include "determinant.h"
#include "excitations.h"
#include "hamiltonian.h"

namespace ipie {

typedef std::unordered_map<ipie::BitString, std::vector<ipie::Determinant>, ipie::BitStringHasher> bs_map;
typedef std::unordered_map<ipie::Determinant, ipie::complex_t, ipie::DeterminantHasher> det_map;

struct Wavefunction {
    // constructors
    Wavefunction(
        ipie::det_map detr_map,
        ipie::bs_map unique_det_a,
        ipie::bs_map unique_det_b,
        ipie::bs_map epq_alpha,
        ipie::bs_map epq_beta);

    static Wavefunction build_wavefunction_from_occ_list(
        std::vector<std::complex<double>> &ci_coeffs,
        std::vector<std::vector<size_t>> &occa,
        std::vector<std::vector<size_t>> &occb,
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
    //
    // member variables
    //
    ipie::det_map dmap;
    ipie::bs_map map_a;
    ipie::bs_map map_b;
    ipie::bs_map epq_a;
    ipie::bs_map epq_b;
    size_t num_dets;
    size_t num_elec;
    size_t num_alpha;
    size_t num_beta;
    size_t num_spatial;
};

}  // namespace ipie

#endif
