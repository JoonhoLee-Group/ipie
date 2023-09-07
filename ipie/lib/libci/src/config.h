#ifndef _IPIE_CONFIG_H
#define _IPIE_CONFIG_H

#include <complex>
#include <tuple>
#include <vector>

namespace ipie {
typedef std::complex<double> complex_t;
typedef std::vector<std::complex<double>> cvec;
typedef std::pair<size_t, size_t> indx_t;
}  // namespace ipie

#endif