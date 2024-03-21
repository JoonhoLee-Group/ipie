#include "wavefunction.h"

#include <random>

#include "gtest/gtest.h"

#include "bitstring.h"
#include "config.h"
#include "determinant.h"
#include "testing.h"

TEST(wavefunction, constructor) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    std::vector<int> occs = {1, 3, 5, 7};
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 5, 9});
    std::vector<std::complex<double>> coeffs = {std::complex<double>(1.0, -1.0), std::complex<double>(1.0, 1.0)};
    std::vector<std::vector<size_t>> occa = {{}, {}};
    std::vector<std::vector<size_t>> occb = {{0, 1, 2, 3}, {0, 1, 2, 4}};
    ipie::Wavefunction wfna = ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occa, occb, 7);
    std::vector<ipie::BitString> dets = {a, b};
    ipie::Wavefunction wfnb = ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occa, occb, 7);
    ASSERT_EQ(wfna, wfnb);
    occb = {{0, 1, 2, 3}, {0, 1, 2, 9}};
    ipie::Wavefunction wfnc = ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occa, occb, 7);
    ASSERT_NE(wfna, wfnc);
}

TEST(determinant, constructor) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b = ipie::BitString(14);
    std::vector<size_t> occs = {1, 3, 5, 7};
    a.set_bits({1, 3, 5, 7});
    b.set_bits({1, 3, 5, 9});
    ipie::Determinant det(a, b);
    ASSERT_EQ(a, det.alpha);
    ASSERT_EQ(b, det.beta);
}

TEST(wavefunction, constructor_maps) {
    ipie::BitString a = ipie::BitString(14);
    ipie::BitString b0 = ipie::BitString(14);
    ipie::BitString b1 = ipie::BitString(14);
    a.set_bits({1, 3, 5, 7});
    b0.set_bits({1, 3, 5, 9});
    std::vector<ipie::Determinant> dets;
    dets.push_back(ipie::Determinant(a, b0));
    b1.set_bits({1, 3, 5, 11});
    dets.push_back(ipie::Determinant(a, b1));
    ASSERT_EQ(dets.size(), 2);
    ipie::bs_map dmap_a;
    dmap_a.insert({a, {dets[0]}});
    auto found = dmap_a.find(a);
    ASSERT_TRUE(found != dmap_a.end());
    dmap_a[a].push_back(dets[1]);
    auto det_ix = dmap_a[dets[0].alpha];
    ASSERT_EQ(det_ix.size(), 2);
    // ASSERT_EQ(dets[det_ix[0]], dets[0]);
    // ASSERT_EQ(dets[det_ix[1]], dets[1]);
    // todo return expected sizes
    auto wfn = ipie::build_test_wavefunction(1000, 10, 5, 5);
    // ASSERT_EQ(wfn.dmap.size(), wfn.coeffs.size());
    // ASSERT_EQ(wfn.num_dets, wfn.coeffs.size());
    size_t ndets = 0;
    ipie::complex_t norm_con;
    auto norm = wfn.norm();
    // std::vector<ipie::complex_t> coeffs_recon(wfn.coeffs.size());
    for (const auto& [key, value] : wfn.map_a) {
        ndets += value.size();
        for (auto i : value) {
            ASSERT_EQ(key, i.alpha);
            ipie::complex_t coeff = wfn.dmap[i];
            norm_con += conj(coeff) * coeff;
        }
    }
    norm_con = sqrt(norm_con);
    // test all the coefficients are there
    // for (size_t i = 0; i < wfn.coeffs.size(); i++) {
    //     ASSERT_DOUBLE_EQ(coeffs_recon[i].real(), wfn.coeffs[i].real());
    //     ASSERT_DOUBLE_EQ(coeffs_recon[i].imag(), wfn.coeffs[i].imag());
    // }
    ASSERT_EQ(ndets, wfn.dmap.size());
    // check we can compute the norm from the unique maps
    EXPECT_NEAR(norm.real(), norm_con.real(), 1e-12);
    EXPECT_NEAR(norm.imag(), norm_con.imag(), 1e-12);
    ndets = 0;
    norm_con = ipie::complex_t{0};
    for (const auto& [key, value] : wfn.map_b) {
        ndets += value.size();
        for (auto i : value) {
            ASSERT_EQ(key, i.beta);
            norm_con += conj(wfn.dmap[i]) * wfn.dmap[i];
        }
    }
    norm_con = sqrt(norm_con);
    EXPECT_NEAR(norm.real(), norm_con.real(), 1e-12);
    EXPECT_NEAR(norm.imag(), norm_con.imag(), 1e-12);
    ASSERT_EQ(ndets, wfn.dmap.size());
    // check that our singles maps only contain single excitations!
    for (const auto& [key, value] : wfn.epq_a) {
        for (auto i : value) {
            ASSERT_EQ(key.count_difference(i.alpha), 1);
        }
    }
    for (const auto& [key, value] : wfn.epq_b) {
        for (auto i : value) {
            ASSERT_EQ(key.count_difference(i.beta), 1);
        }
    }
    auto cis = ipie::build_ci_singles(20, 5, 5);
    size_t oava = cis.num_alpha * (cis.num_spatial - cis.num_alpha);
    size_t obvb = cis.num_beta * (cis.num_spatial - cis.num_beta);
    ASSERT_EQ(oava + obvb + 1, cis.num_dets);
    ipie::BitString aa(cis.num_spatial);
    for (size_t i = 0; i < cis.num_alpha; i++) {
        aa.set_bit(i);
    }
    ASSERT_EQ(cis.epq_a[aa].size(), oava);
    ipie::BitString bb(cis.num_spatial);
    for (size_t i = 0; i < cis.num_beta; i++) {
        bb.set_bit(i);
    }
    ASSERT_EQ(cis.epq_b[bb].size(), obvb);
}

TEST(wavefunction, density_matrix_restricted) {
    auto wfn = ipie::build_test_wavefunction(200, 10, 5, 5);
    auto dm = wfn.build_one_rdm();
    ipie::complex_t trace{0.0};
    for (size_t p = 0; p < wfn.num_spatial; p++) {
        trace += dm[p * wfn.num_spatial + p];
    }
    ASSERT_NEAR(trace.real(), wfn.num_alpha, 1e-12);
    ASSERT_NEAR(trace.imag(), 0.0, 1e-12);
    trace = 0;
    for (size_t p = 0; p < wfn.num_spatial; p++) {
        trace += dm[p * wfn.num_spatial + p + wfn.num_spatial * wfn.num_spatial];
    }
    ASSERT_NEAR(trace.real(), wfn.num_beta, 1e-12);
    ASSERT_NEAR(trace.imag(), 0.0, 1e-12);
}

TEST(wavefunction, density_matrix_polarized) {
    auto wfn = ipie::build_test_wavefunction(200, 100, 3, 2);
    auto dm = wfn.build_one_rdm();
    ipie::complex_t trace{0.0};
    for (size_t p = 0; p < wfn.num_spatial; p++) {
        trace += dm[p * wfn.num_spatial + p];
    }
    ASSERT_NEAR(trace.real(), wfn.num_alpha, 1e-12);
    ASSERT_NEAR(trace.imag(), 0.0, 1e-12);
    trace = 0;
    for (size_t p = 0; p < wfn.num_spatial; p++) {
        trace += dm[p * wfn.num_spatial + p + wfn.num_spatial * wfn.num_spatial];
    }
    ASSERT_NEAR(trace.real(), wfn.num_beta, 1e-12);
    ASSERT_NEAR(trace.imag(), 0.0, 1e-12);
}

// TEST(wavefunction, profile) {
//     std::vector<int> ndet = {100, 1000, 10000};
//     for (auto i : ndet) {
//         auto wfn = ipie::build_test_wavefunction(i, 12, 6, 6);
//         auto t1 = std::chrono::high_resolution_clock::now();
//         auto dm = wfn.build_one_rdm();
//         auto t2 = std::chrono::high_resolution_clock::now();
//         std::chrono::duration<double, std::milli> ms_double = t2 - t1;
//         std::cout << "ndet: " << i << " " << wfn.num_dets << " " << ms_double.count() / 1000 << std::endl;
//     }
// }
// write a test that generates a uniform distribution and compute the appropriate probabilities.

// TEST(wavefunction, variational_energy) {
//     for (int i = 0; i < 5; i++) {
//         auto wfn = ipie::build_test_wavefunction_restricted(10, 20);
//         auto ham = ipie::build_test_hamiltonian(wfn.num_spatial);
//         auto energy = wfn.energy(ham);
//         ASSERT_NEAR(energy.etot.real(), energy.e1b.real() + energy.e2b.real(), 1e-12);
//         ASSERT_NEAR(energy.etot.imag(), energy.e1b.imag() + energy.e2b.imag(), 1e-12);
//     }
// }