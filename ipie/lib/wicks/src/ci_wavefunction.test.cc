#include "ci_wavefunction.h"

#include "gtest/gtest.h"

#include "bitstring.h"

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