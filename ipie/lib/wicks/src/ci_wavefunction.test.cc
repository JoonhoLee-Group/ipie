#include "ci_wavefunction.h"

#include <random>

#include "gtest/gtest.h"

#include "bitstring.h"
#include "config.h"

std::pair<std::vector<ipie::complex_t>, std::vector<ipie::complex_t>> build_test_hamiltonian(size_t num_orb) {
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
    return std::make_pair(h1e, h2e);
}

TEST(ci_wavefunction, density_matrix) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    std::vector<int> occs = {1, 3, 5, 7};
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 5, 7});
    std::vector<ipie::BitString> dets = {a, b};
    std::vector<std::complex<double>> coeffs = {std::complex<double>(1.0, -1.0), std::complex<double>(1.0, 1.0)};
    ipie::CIWavefunction wfn(coeffs, dets);
    auto dm = wfn.build_one_rdm(2);
    for (size_t p = 0; p < 7; p++) {
        for (size_t q = 0; q < 7; q++) {
            ASSERT_EQ(dm[p * 7 + q], 0.0);
        }
    }
    for (size_t p = 0; p < 7; p++) {
        for (size_t q = 0; q < 7; q++) {
            if (p == q && std::find(occs.begin(), occs.end(), 2 * p + 1) != occs.end()) {
                ASSERT_EQ(dm[p * 7 + q + 7 * 7], 1.0);
            } else {
                ASSERT_EQ(dm[p * 7 + q + 7 * 7], 0.0);
            }
        }
    }
    a.clear_bits();
    b.clear_bits();
    a.set_bits({1, 3, 5, 7});
    b.set_bits({0, 2, 4, 6});
    // <uuuu|p^q|uuuu> + <dddd|p^q|uuuu>
    dets = {a, b};
    ipie::CIWavefunction wfn2(coeffs, dets);
    dm = wfn2.build_one_rdm(2);
    std::vector<int> occsa = {0, 2, 4, 6};
    for (size_t p = 0; p < 7; p++) {
        for (size_t q = 0; q < 7; q++) {
            if (p == q && std::find(occsa.begin(), occsa.end(), 2 * p) != occsa.end()) {
                ASSERT_EQ(dm[p * 7 + q], 0.5);
            } else {
                ASSERT_EQ(dm[p * 7 + q], 0.0);
            }
        }
    }
    std::vector<int> occsb = {1, 3, 5, 7};
    for (size_t p = 0; p < 7; p++) {
        for (size_t q = 0; q < 7; q++) {
            if (p == q && std::find(occsb.begin(), occsb.end(), 2 * p + 1) != occsb.end()) {
                ASSERT_EQ(dm[p * 7 + q + 7 * 7], 0.5);
            } else {
                ASSERT_EQ(dm[p * 7 + q + 7 * 7], 0.0);
            }
        }
    }
}

TEST(ci_wavefunction, slater_condon0) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    std::vector<int> occs = {1, 3, 5, 7};
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 5, 7});
    std::vector<ipie::BitString> dets = {a, b};
    std::vector<std::complex<double>> coeffs = {std::complex<double>(1.0, -1.0), std::complex<double>(1.0, 1.0)};
    ipie::CIWavefunction wfn(coeffs, dets);
    auto ham = build_test_hamiltonian(a.num_bits);
    auto e0 = wfn.slater_condon0(occs, ham.first, ham.second);
    ASSERT_NEAR(std::real(std::get<0>(e0)), std::real(std::get<1>(e0) + std::get<2>(e0)), 1e-12);
}

TEST(ci_wavefunction, slater_condon1) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 5, 9});
    std::vector<ipie::BitString> dets = {a, b};
    std::vector<std::complex<double>> coeffs = {std::complex<double>(1.0, -1.0), std::complex<double>(1.0, 1.0)};
    auto ham = build_test_hamiltonian(a.num_bits);
    ipie::CIWavefunction wfn(coeffs, dets);
    ipie::Excitation excit_ia(1);
    decode_single_excitation(a, b, excit_ia);
    std::vector<int> occs = {1, 3, 5, 7};
    auto e1 = wfn.slater_condon1(occs, excit_ia, ham.first, ham.second);
    ASSERT_NEAR(std::real(std::get<0>(e1)), std::real(std::get<1>(e1) + std::get<2>(e1)), 1e-12);
}

TEST(ci_wavefunction, slater_condon2) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 9, 11});
    std::vector<ipie::BitString> dets = {a, b};
    std::vector<std::complex<double>> coeffs = {std::complex<double>(1.0, -1.0), std::complex<double>(1.0, 1.0)};
    auto ham = build_test_hamiltonian(a.num_bits);
    ipie::CIWavefunction wfn(coeffs, dets);
    ipie::Excitation excit_ijab(1);
    decode_single_excitation(a, b, excit_ijab);
    auto e2 = wfn.slater_condon2(excit_ijab, ham.second);
    ASSERT_NEAR(std::real(std::get<0>(e2)), std::real(std::get<2>(e2)), 1e-12);
}