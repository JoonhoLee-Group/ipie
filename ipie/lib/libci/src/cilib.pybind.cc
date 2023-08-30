#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "config.h"
#include "observables.h"
#include "wavefunction.h"

namespace py = pybind11;

PYBIND11_MODULE(libci, m) {
    m.doc() = "A lightweight ci utility library for computing properties of CI-like wavefunctions.";
    m.def(
        "one_rdm",
        // this works but is not very idiomatic as we expect numpy arrays as input. it works because of the stl wrapper,
        // but should probably cast internally.
        [](std::vector<std::complex<double>> &coeffs,
           std::vector<std::vector<int>> &occa,
           std::vector<std::vector<int>> &occb,
           py::ssize_t num_spatial) {
            ipie::Wavefunction wfn(coeffs, occa, occb, num_spatial);
            // std::cout << "\n" << wfn << std::endl;
            std::vector<ipie::complex_t> opdm = ipie::build_one_rdm(wfn);
            // https://github.com/pybind/pybind11/issues/1299
            // need to check if this is problematic with ownership.
            // I can live with copying given the array is so small...
            return py::array_t<ipie::complex_t>(
                std::vector<ptrdiff_t>{2, (py::ssize_t)num_spatial, (py::ssize_t)num_spatial}, &opdm[0]);
        },
        "Compute the one-particle reduced density matrix.");
    // m.def("test", &test_function, "Compute the one-particle reduced density matrix.");
    // m.def("variational_energy_wrap", &ipie::variational_energy, "Compute the one-particle reduced density matrix.");
}