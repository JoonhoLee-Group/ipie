#include "ci_wavefunction.h"

#include "gtest/gtest.h"

#include "bitstring.h"

TEST(ci_wavefunction, density_matrix) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 5, 7});
    std::vector<ipie::BitString> dets = {a, b};
    std::vector<std::complex<double>> coeffs = {std::complex<double>(1.0, -1.0), std::complex<double>(1.0, 1.0)};
    ipie::CIWavefunction wfn(coeffs, dets);
    auto dm = wfn.build_one_rdm(2);
    ASSERT_EQ(dm[7 * 7], 1.0);
    ASSERT_EQ(dm[1 * 7 + 1 + 7 * 7], 1.0);
    ASSERT_EQ(dm[2 * 7 + 2 + 7 * 7], 1.0);
    ASSERT_EQ(dm[3 * 7 + 3 + 7 * 7], 1.0);
}