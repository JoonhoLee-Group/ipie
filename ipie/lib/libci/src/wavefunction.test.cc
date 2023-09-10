#include "wavefunction.h"

#include <random>

#include "gtest/gtest.h"

#include "bitstring.h"
#include "config.h"
#include "testing.h"

TEST(wavefunction, constructor) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    std::vector<int> occs = {1, 3, 5, 7};
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 5, 9});
    std::vector<std::complex<double>> coeffs = {std::complex<double>(1.0, -1.0), std::complex<double>(1.0, 1.0)};
    std::vector<std::vector<int>> occa = {{}, {}};
    std::vector<std::vector<int>> occb = {{0, 1, 2, 3}, {0, 1, 2, 4}};
    ipie::Wavefunction wfna = ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occa, occb, 7);
    std::vector<ipie::BitString> dets = {a, b};
    ipie::Wavefunction wfnb = ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occa, occb, 7);
    ASSERT_EQ(wfna, wfnb);
    occb = {{0, 1, 2, 3}, {0, 1, 2, 9}};
    ipie::Wavefunction wfnc = ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occa, occb, 7);
    ASSERT_NE(wfna, wfnc);
}

TEST(wavefunction, constructor_map) {
    for (int i = 0; i < 10; i++) {
        auto wfn = ipie::build_test_wavefunction(1000, 100);
        for (auto const& pair : wfn.map) {
            ASSERT_EQ(wfn.map[pair.first], pair.second);
        }
    }
}

TEST(wavefunction, density_matrix_restricted) {
    auto wfn = ipie::build_test_wavefunction_restricted(100, 50);
    auto dm = wfn.build_one_rdm();
    ipie::complex_t trace{0.0};
    for (size_t p = 0; p < wfn.num_spatial; p++) {
        trace += dm[p * wfn.num_spatial + p];
    }
    ASSERT_NEAR(trace.real(), wfn.num_elec / 2, 1e-12);
    trace = 0;
    for (size_t p = 0; p < wfn.num_spatial; p++) {
        trace += dm[p * wfn.num_spatial + p + wfn.num_spatial * wfn.num_spatial];
    }
    ASSERT_NEAR(trace.real(), wfn.num_elec / 2, 1e-12);
    ASSERT_NEAR(trace.imag(), 0.0, 1e-12);
}

TEST(wavefunction, density_matrix_polarized) {
    auto wfn = ipie::build_test_wavefunction(1000, 50);
    auto dm = wfn.build_one_rdm();
    ipie::complex_t trace{0.0};
    for (size_t p = 0; p < wfn.num_spatial; p++) {
        trace += dm[p * wfn.num_spatial + p];
    }
    for (size_t p = 0; p < wfn.num_spatial; p++) {
        trace += dm[p * wfn.num_spatial + p + wfn.num_spatial * wfn.num_spatial];
    }
    ASSERT_NEAR(trace.real(), wfn.num_elec, 1e-12);
    ASSERT_NEAR(trace.imag(), 0.0, 1e-12);
}
