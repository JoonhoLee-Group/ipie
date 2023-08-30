#include <pybind11/complex.h>
#include <pybind11/iostream.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

#include "config.h"
#include "hamiltonian.h"
#include "observables.h"
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
    m.def(
        "one_rdm",
        // this works but is not very idiomatic as we expect numpy arrays as input. it works because of the stl wrapper,
        // but should probably cast internally.
        [](std::vector<std::complex<double>> &coeffs,
           std::vector<std::vector<int>> &occa,
           std::vector<std::vector<int>> &occb,
           size_t num_spatial) {
            ipie::Wavefunction wfn =
                ipie::Wavefunction::build_wavefunction_from_occ_list(coeffs, occa, occb, num_spatial);
            // std::cout << "\n" << wfn << std::endl;
            std::vector<ipie::complex_t> opdm = ipie::build_one_rdm(wfn);
            // https://github.com/pybind/pybind11/issues/1299
            // need to check if this is problematic with ownership.
            // I can live with copying given the array is so small...
            return py::array_t<ipie::complex_t>(
                std::vector<ptrdiff_t>{2, (py::ssize_t)num_spatial, (py::ssize_t)num_spatial}, &opdm[0]);
        },
        "Compute the one-particle reduced density matrix.");
    m.def("variational_energy", &ipie::compute_variational_energy, "Compute the variational energy.");

    py::class_<ipie::Hamiltonian>(m, "Hamiltonian")
        .def(py::init(&build_hamiltonian_from_numpy_ndarray))
        .def_readonly("num_spatial", &ipie::Hamiltonian::num_spatial)
        .def_readonly("h1e", &ipie::Hamiltonian::h1e)
        .def_readonly("h2e", &ipie::Hamiltonian::h2e);

    py::class_<ipie::Wavefunction>(m, "Wavefunction")
        .def(py::init(&ipie::Wavefunction::build_wavefunction_from_occ_list))
        .def("norm", &ipie::Wavefunction::norm)
        .def_readonly("num_dets", &ipie::Wavefunction::num_dets)
        .def_readonly("num_elec", &ipie::Wavefunction::num_elec)
        .def_readonly("num_spatial", &ipie::Wavefunction::num_spatial);

    // these are just for testing purposes so ignore inneficiencies.
    m.def(
        "slater_condon0",
        [](py::array_t<ipie::complex_t> &h1e,
           py::array_t<ipie::complex_t> &h2e,
           ipie::complex_t e0,
           py::array_t<int> &occs) {
            auto ham = build_hamiltonian_from_numpy_ndarray(h1e, h2e, e0);
            std::vector<int> _occs(occs.data(), occs.data() + occs.size());
            return ipie::slater_condon0(ham, _occs);
        });
    m.def(
        "slater_condon1",
        [](py::array_t<ipie::complex_t> &h1e,
           py::array_t<ipie::complex_t> &h2e,
           py::array_t<int> &occs,
           py::ssize_t i,
           py::ssize_t a) {
            auto ham = build_hamiltonian_from_numpy_ndarray(h1e, h2e, ipie::complex_t{0});
            std::vector<int> _occs(occs.data(), occs.data() + occs.size());
            auto excit_ia = ipie::Excitation(1);
            excit_ia.from[0] = i;
            excit_ia.to[0] = a;
            return ipie::slater_condon1(ham, _occs, excit_ia);
        });
    m.def(
        "slater_condon2",
        [](py::array_t<ipie::complex_t> &h2e, py::ssize_t i, py::ssize_t j, py::ssize_t a, py::ssize_t b) {
            py::array_t<ipie::complex_t> h1e;
            auto ham = build_hamiltonian_from_numpy_ndarray(h1e, h2e, ipie::complex_t{0});
            auto excit_ijab = ipie::Excitation(2);
            excit_ijab.from = {(size_t)i, (size_t)j};
            excit_ijab.to = {(size_t)a, (size_t)b};
            return ipie::slater_condon2(ham, excit_ijab);
        });
}