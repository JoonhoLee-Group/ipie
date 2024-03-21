#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "config.h"
#include "hamiltonian.h"
#include "wavefunction.h"

namespace py = pybind11;

ipie::Hamiltonian build_hamiltonian_from_numpy_ndarray(
    py::array_t<ipie::complex_t> &h1e, py::array_t<ipie::complex_t> &h2e, ipie::complex_t e0) {
    // sometimes we pass in a zero array for either tensors (but not both).
    py::ssize_t num_spatial = std::max(h1e.shape(0), h2e.shape(1));
    std::vector<ipie::complex_t> _h1e(h1e.data(), h1e.data() + h1e.size());
    std::vector<ipie::complex_t> _h2e(h2e.data(), h2e.data() + h2e.size());
    return ipie::Hamiltonian(_h1e, _h2e, e0, num_spatial);
}

PYBIND11_MODULE(libci, m) {
    m.doc() = "A lightweight ci utility library for computing properties of CI-like wavefunctions.";

    py::class_<ipie::Hamiltonian>(m, "Hamiltonian")
        .def(py::init(&build_hamiltonian_from_numpy_ndarray))
        .def_readonly("num_spatial", &ipie::Hamiltonian::num_spatial)
        .def_readonly("h1e", &ipie::Hamiltonian::h1e)
        .def_readonly("h2e", &ipie::Hamiltonian::h2e);

    py::class_<ipie::Wavefunction>(m, "Wavefunction")
        .def(py::init(&ipie::Wavefunction::build_wavefunction_from_occ_list))
        .def("norm", &ipie::Wavefunction::norm)
        .def(
            "energy",
            [](ipie::Wavefunction &self, ipie::Hamiltonian &ham) {
                auto etot = self.energy(ham);
                return std::vector<ipie::complex_t>{etot.etot, etot.e1b, etot.e2b};
            })
        .def("one_rdm", &ipie::Wavefunction::build_one_rdm)
        .def_readonly("num_dets", &ipie::Wavefunction::num_dets)
        .def_readonly("num_elec", &ipie::Wavefunction::num_elec)
        .def_readonly("num_spatial", &ipie::Wavefunction::num_spatial);
}