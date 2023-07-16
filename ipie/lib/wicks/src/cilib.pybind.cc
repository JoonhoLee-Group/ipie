#include <pybind11/iostream.h>
#include <pybind11/pybind11.h>

pybind11::class_<DemSampler> stim_pybind::pybind_ci_wfn(pybind11::module &m) {
    return pybind11::class_<CIWavefunction>(
        m,
        "CIWavefunction",
        clean_doc_string(R"DOC(
    )DOC")
            .data());
}

void ipie_pybind::pybind_ci_wavefunction_method(pybind11::module &m, pybind11::class_<ipie::CIWavefunction> &w) {
    c.def(
        "one_rdm",
        [](CIWavefunction &self, size_t num_dets) {
            return self.build_one_rdm(num_dets);
        },
        pybind11::kw_only(),
        pybind11::arg("num_dets") = pybind11::none(),
        clean_doc_string(R"DOC(
    )DOC")
            .data());
    c.def(
        "variational_energy",
        [](CIWavefunction &self, size_t num_dets) {
            return self.variational_energy(num_dets);
        },
        pybind11::kw_only(),
        pybind11::arg("num_dets") = pybind11::none(),
        clean_doc_string(R"DOC(
    )DOC")
            .data());
}

PYBIND11_MODULE(cilib, m) {
    m.doc() = R"pbdoc(
        A lightweight ci utility library for computing properties of CI-like
        wavefunctions.
    )pbdoc";
    auto ci_wfn = pybind_ci_wfn(m);
    pybind_ci_wavefunction_methods(m, ci_wfn);
}