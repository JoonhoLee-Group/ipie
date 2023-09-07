#include "hamiltonian.h"

#include <random>

#include "gtest/gtest.h"

#include "testing.h"

using namespace ipie;
TEST(hamiltonian, constructor) {
    auto ham = build_test_hamiltonian(14);
    ASSERT_EQ(ham.h1e.size(), 14 * 14);
    ASSERT_EQ(ham.h2e.size(), 14 * 14 * 14 * 14);
}

TEST(hamiltonian, slater_condon0) {
    std::vector<int> occs = {1, 3, 5, 7};
    auto ham = build_test_hamiltonian(14);
    auto e0 = slater_condon0(ham, occs);
    ASSERT_NEAR(e0.etot.real(), e0.e1b.real() + e0.e2b.real(), 1e-12);
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
    ASSERT_NEAR(e1.etot.real(), e1.e1b.real() + e1.e2b.real(), 1e-12);
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
    ASSERT_NEAR(0.0, e2.e1b.real(), 1e-12);
    ASSERT_NEAR(e2.etot.real(), e2.e2b.real(), 1e-12);
}