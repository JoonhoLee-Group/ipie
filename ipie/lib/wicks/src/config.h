#ifndef _IPIE_CONFIG_H
#define _IPIE_CONFIG_H

#include <complex>
#include <vector>

namespace ipie {
typedef std::complex<double> complex_t;
typedef std::vector<std::complex<double>> cvec;
typedef std::tuple<complex_t, complex_t, complex_t> energy_t;
typedef std::pair<size_t, size_t> indx_t;
}  // namespace ipie

#endif