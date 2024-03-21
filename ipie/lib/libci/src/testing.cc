#include "testing.h"

#include <algorithm>
#include <chrono>
#include <random>

#include "bitstring.h"
#include "determinant.h"

namespace ipie {
ipie::Hamiltonian build_test_hamiltonian(size_t num_orb) {
    std::vector<ipie::complex_t> h1e(num_orb * num_orb);
    std::vector<ipie::complex_t> h2e(num_orb * num_orb * num_orb * num_orb);
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0, 1.0);
    for (size_t i = 0; i < h1e.size(); i++) {
        h1e[i] = ipie::complex_t(distribution(generator), distribution(generator));
    }
    for (size_t i = 0; i < h2e.size(); i++) {
        h2e[i] = ipie::complex_t(distribution(generator), distribution(generator));
    }
    ipie::complex_t e0 = ipie::complex_t(distribution(generator), 0.0);
    return ipie::Hamiltonian(h1e, h2e, e0, num_orb);
}

ipie::Wavefunction build_test_wavefunction(
    size_t num_dets_max, size_t num_spat_max, size_t num_alpha, size_t num_beta) {
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> det_dist(1, num_dets_max);
    size_t ndets = (size_t)det_dist(generator);
    std::uniform_int_distribution<int> spat_dist(std::max(num_alpha, num_beta), num_spat_max);
    std::uniform_real_distribution<double> c_dist(0, 1);
    int num_spin = 2 * spat_dist(generator);
    ipie::bs_map dmap;
    std::vector<size_t> range(num_spin / 2);
    std::vector<std::vector<size_t>> occs_a;
    std::vector<std::vector<size_t>> occs_b;
    std::vector<ipie::complex_t> coeffs;
    for (size_t idet = 0; idet < ndets; idet++) {
        std::iota(range.begin(), range.end(), 0);
        std::shuffle(range.begin(), range.end(), generator);
        std::vector<size_t> occa(num_alpha);
        std::copy(range.begin(), range.begin() + num_alpha, occa.begin());
        occs_a.push_back(occa);
        std::iota(range.begin(), range.end(), 0);
        std::vector<size_t> occb(num_alpha);
        std::shuffle(range.begin(), range.end(), generator);
        std::copy(range.begin(), range.begin() + num_beta, occb.begin());
        occs_b.push_back(occb);
        std::complex<double> coeff = std::complex<double>(c_dist(generator), 0.0);
        coeffs.push_back(coeff);
    }
    // remove any duplicates, very silly but only a test utility function so who cares.
    std::vector<Determinant> dets_dedupe;
    std::vector<std::vector<size_t>> occs_a_dedupe;
    std::vector<std::vector<size_t>> occs_b_dedupe;
    std::vector<ipie::complex_t> coeffs_dedupe;
    for (size_t i = 0; i < coeffs.size(); i++) {
        ipie::BitString a(num_spin / 2);
        a.set_bits(occs_a[i]);
        ipie::BitString b(num_spin / 2);
        b.set_bits(occs_b[i]);
        ipie::Determinant new_det(a, b);
        if (i == 0) {
            dets_dedupe.push_back(new_det);
            occs_a_dedupe.push_back(occs_a[i]);
            occs_b_dedupe.push_back(occs_b[i]);
            coeffs_dedupe.push_back(coeffs[i]);
        } else if (std::find(dets_dedupe.begin(), dets_dedupe.end(), new_det) == dets_dedupe.end()) {
            dets_dedupe.push_back(new_det);
            occs_a_dedupe.push_back(occs_a[i]);
            occs_b_dedupe.push_back(occs_b[i]);
            coeffs_dedupe.push_back(coeffs[i]);
        }
    }
    ipie::Wavefunction wfn =
        ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs_dedupe, occs_a_dedupe, occs_b_dedupe, num_spin / 2);
    return wfn;
}
ipie::Wavefunction build_ci_singles(size_t num_spat_max, size_t num_alpha, size_t num_beta) {
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> spat_dist(std::max(num_alpha, num_beta), num_spat_max);
    std::uniform_real_distribution<double> c_dist(0, 1);
    int num_spin = 2 * spat_dist(generator);
    std::vector<size_t> range(num_spin / 2);
    std::vector<std::vector<size_t>> occs_a;
    std::vector<std::vector<size_t>> occs_b;
    std::vector<ipie::complex_t> coeffs;
    // build ci singles wavefunction
    ipie::BitString a(num_spin / 2);
    ipie::BitString b(num_spin / 2);
    for (size_t i = 0; i < num_alpha; i++) {
        a.set_bit(i);
    }
    for (size_t i = 0; i < num_beta; i++) {
        b.set_bit(i);
    }
    std::vector<std::vector<size_t>> occsa;
    std::vector<std::vector<size_t>> occsb;
    std::vector<size_t> ref_occ_a(num_alpha);
    std::vector<size_t> ref_occ_b(num_beta);
    std::iota(ref_occ_a.begin(), ref_occ_a.end(), 0);
    std::iota(ref_occ_b.begin(), ref_occ_b.end(), 0);
    coeffs.push_back(ipie::complex_t{1.0});
    occsa.push_back(ref_occ_a);
    occsb.push_back(ref_occ_b);
    for (size_t o = 0; o < num_alpha; o++) {
        a.clear_bit(o);
        std::vector<size_t> occa(num_alpha);
        std::vector<size_t> occb(num_beta);
        for (size_t p = num_alpha; p < (size_t)(num_spin / 2); p++) {
            a.set_bit(p);
            a.decode_bits(occa);
            occsa.push_back(occa);
            occsb.push_back(ref_occ_b);
            coeffs.push_back(ipie::complex_t{1.0});
            a.clear_bit(p);
        }
        a.set_bit(o);
    }
    for (size_t o = 0; o < num_beta; o++) {
        b.clear_bit(o);
        std::vector<size_t> occa(num_alpha);
        std::vector<size_t> occb(num_beta);
        for (size_t p = num_beta; (size_t)(p < num_spin / 2); p++) {
            b.set_bit(p);
            b.decode_bits(occb);
            occsa.push_back(ref_occ_a);
            occsb.push_back(occb);
            coeffs.push_back(ipie::complex_t{1.0});
            b.clear_bit(p);
        }
        b.set_bit(o);
    }
    ipie::Wavefunction wfn = ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occsa, occsb, num_spin / 2);
    return wfn;
}
}  // namespace ipie