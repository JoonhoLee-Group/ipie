#include "observables.h"

#include <complex>

#include "gtest/gtest.h"

#include "bitstring.h"
#include "hamiltonian.h"
#include "random"
#include "testing.h"
#include "wavefunction.h"

TEST(observables, density_matrix_restricted) {
    auto wfn = ipie::build_test_wavefunction_restricted(100, 50);
    auto dm = ipie::build_one_rdm(wfn);
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

TEST(observables, density_matrix_polarized) {
    auto wfn = ipie::build_test_wavefunction(1000, 50);
    auto dm = ipie::build_one_rdm(wfn);
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

TEST(observables, variational_energy) {
    for (int i = 0; i < 5; i++) {
        auto wfn = ipie::build_test_wavefunction_restricted(10, 20);
        auto ham = ipie::build_test_hamiltonian(wfn.num_spatial);
        auto energy = compute_variational_energy(wfn, ham);
        ASSERT_NEAR(energy.etot.real(), energy.e1b.real() + energy.e2b.real(), 1e-12);
        ASSERT_NEAR(energy.etot.imag(), energy.e1b.imag() + energy.e2b.imag(), 1e-12);
    }
}