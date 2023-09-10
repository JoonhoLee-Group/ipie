#include "observables.h"

#include <complex>

#include "gtest/gtest.h"

#include "bitstring.h"
#include "hamiltonian.h"
#include "random"
#include "testing.h"
#include "wavefunction.h"

TEST(observables, variational_energy) {
    for (int i = 0; i < 5; i++) {
        auto wfn = ipie::build_test_wavefunction_restricted(10, 20);
        auto ham = ipie::build_test_hamiltonian(wfn.num_spatial);
        auto energy = compute_variational_energy(wfn, ham);
        ASSERT_NEAR(energy.etot.real(), energy.e1b.real() + energy.e2b.real(), 1e-12);
        ASSERT_NEAR(energy.etot.imag(), energy.e1b.imag() + energy.e2b.imag(), 1e-12);
    }
}