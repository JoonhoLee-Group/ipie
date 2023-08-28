#include "wavefunction.h"

#include <random>

#include "gtest/gtest.h"

#include "bitstring.h"
#include "config.h"

TEST(wavefunction, constructor) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    std::vector<int> occs = {1, 3, 5, 7};
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 5, 7});
    std::vector<std::complex<double>> coeffs = {std::complex<double>(1.0, -1.0), std::complex<double>(1.0, 1.0)};
    std::vector<std::vector<int>> occa = {{}, {}};
    std::vector<std::vector<int>> occb = {{0, 1, 2, 3}, {0, 1, 2, 3}};
    ipie::Wavefunction wfna(coeffs, occa, occb, 7);
    std::vector<ipie::BitString> dets = {a, b};
    ipie::Wavefunction wfnb(coeffs, dets);
    ASSERT_EQ(wfna, wfnb);
}
