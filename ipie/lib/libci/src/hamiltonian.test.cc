#include "hamiltonian.h"

#include <random>

#include "gtest/gtest.h"

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

TEST(hamiltonian, constructor) {
    auto ham = build_test_hamiltonian(14);
    ASSERT_EQ(ham.h1e.size(), 14 * 14);
    ASSERT_EQ(ham.h2e.size(), 14 * 14 * 14 * 14);
}

TEST(hamiltonian, slater_condon0) {
    std::vector<int> occs = {1, 3, 5, 7};
    auto ham = build_test_hamiltonian(14);
    auto e0 = slater_condon0(ham, occs);
    ASSERT_NEAR(std::real(std::get<0>(e0)), std::real(std::get<1>(e0) + std::get<2>(e0)), 1e-12);
}

TEST(hamiltonian, slater_condon1) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    std::vector<int> occs = {1, 3, 5, 7};
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 5, 9});
    auto ham = build_test_hamiltonian(a.num_bits);
    ipie::Excitation excit_ia(1);
    decode_double_excitation(a, b, excit_ia);
    auto e1 = slater_condon1(ham, occs, excit_ia);
    ASSERT_NEAR(std::real(std::get<0>(e1)), std::real(std::get<1>(e1) + std::get<2>(e1)), 1e-12);
}

TEST(hamiltonian, slater_condon2) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 9, 11});
    std::vector<ipie::BitString> dets = {a, b};
    auto ham = build_test_hamiltonian(a.num_bits);
    ipie::Excitation excit_ijab(2);
    decode_double_excitation(a, b, excit_ijab);
    auto e2 = slater_condon2(ham, excit_ijab);
    ASSERT_NEAR(0.0, std::real(std::get<1>(e2)), 1e-12);
    ASSERT_NEAR(std::real(std::get<0>(e2)), std::real(std::get<2>(e2)), 1e-12);
}