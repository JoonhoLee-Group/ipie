#include "observables.h"

#include <complex>

#include "gtest/gtest.h"

#include "bitstring.h"
#include "hamiltonian.h"
#include "random"
#include "testing.h"
#include "wavefunction.h"

TEST(observables, density_matrix) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    std::vector<int> occs = {1, 3, 5, 7};
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 5, 7});
    std::vector<ipie::BitString> dets = {a, b};
    std::vector<std::complex<double>> coeffs = {std::complex<double>(1.0, -1.0), std::complex<double>(1.0, 1.0)};
    ipie::Wavefunction wfn(coeffs, dets);
    auto dm = build_one_rdm(wfn);
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
    ipie::Wavefunction wfn2(coeffs, dets);
    dm = build_one_rdm(wfn2);
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

TEST(observables, density_matrix_alt_constructor) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    std::vector<int> occs = {1, 3, 5, 7};
    std::vector<std::complex<double>> coeffs = {std::complex<double>(1.0, -1.0), std::complex<double>(1.0, 1.0)};
    std::vector<std::vector<int>> occa = {{}, {}};
    std::vector<std::vector<int>> occb = {{0, 1, 2, 3}, {0, 1, 2, 3}};
    ipie::Wavefunction wfn = ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occa, occb, 7);
    auto dm = build_one_rdm(wfn);
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
    // <uuuu|p^q|uuuu> + <dddd|p^q|uuuu>
    occa = {{}, {0, 1, 2, 3}};
    occb = {{0, 1, 2, 3}, {}};
    ipie::Wavefunction wfn2 = ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occa, occb, 7);
    dm = build_one_rdm(wfn2);
    std::vector<int> occsa = {0, 2, 4, 6};
    a.set_bits({1, 3, 5, 7});
    b.set_bits({0, 2, 4, 6});
    std::vector<ipie::BitString> dets = {a, b};
    ipie::Wavefunction wfnb(coeffs, dets);
    ASSERT_EQ(wfnb, wfn2);
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

TEST(observables, variational_energy) {
    auto wfn = ipie::build_test_wavefunction(100, 12);
    auto ham = ipie::build_test_hamiltonian(wfn.num_spatial);
    auto energy = compute_variational_energy(wfn, ham);
    ASSERT_NEAR(energy.etot.real(), energy.e1b.real() + energy.e2b.real(), 1e-12);
    ASSERT_NEAR(energy.etot.imag(), energy.e1b.imag() + energy.e2b.imag(), 1e-12);
}