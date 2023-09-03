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
    ipie::Wavefunction wfna = ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occa, occb, 7);
    std::vector<ipie::BitString> dets = {a, b};
    ipie::Wavefunction wfnb(coeffs, dets);
    ASSERT_EQ(wfna, wfnb);
}

TEST(wavefunction, constructor_map) {
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> det_dist(0, 10000 - 1);
    int ndets = det_dist(generator);
    std::uniform_int_distribution<int> bit_dist(0, 100 - 1);
    std::uniform_real_distribution<double> c_dist(0, 1);
    int num_set = bit_dist(generator);
    std::vector<ipie::BitString> dets;
    std::vector<std::complex<double>> coeffs;
    ipie::det_map dmap;
    for (size_t idet = 0; idet < ndets; idet++) {
        auto a = ipie::BitString(100);
        for (size_t i = 0; i < num_set; i++) {
            a.set_bit((size_t)bit_dist(generator));
        }
        std::complex<double> coeff = std::complex<double>(c_dist(generator), 0.0);
        if (dmap.size() == 0 || dmap.find(a) == dmap.end()) {
            // we don't want duplicate keys.
            dmap.insert({a, coeff});
            dets.push_back(a);
            coeffs.push_back(coeff);
        }
    }
    ipie::Wavefunction wfn(dmap);
    for (size_t idet = 0; idet < ndets; idet++) {
        ASSERT_EQ(coeffs[idet], wfn.map[dets[idet]]);
    }
}