# ```libci development guide```

Light C++ implementations for computing properties (energy and one-rdm) of
CI-like wavefunctions. The main principles is that it should be "fast enough"
for the purposes of AFQMC. In practice this means we want to compute these
quantities for ~ $10^6$ determinants and not be limited by any 64-bit issues. As
such this library should support wavefunctions with an arbitrary number of
electrons and orbitals and is agnostic to their source.

Ideally the user should not have to interact with this library.

For interface purposes we expect the wavefunction to be provided as a list of
determinants which are specified as

$$
|\psi\rangle = \sum_I c_I |D_I^{\alpha}\rangle|D_I^\beta\rangle
$$

where $c_I$ is a complex double precision number and $D_I$ is a list of occupied
**spatial** orbitals. As AFQMC assumes an aaaabbbb ordering for creation and
annihilation operators we also assume this phase convention.

## ```Building with bazel```

Build the library

``` bash
bazel build:libci
```

Run the tests:

``` bash
bazel build:libci_test
./bazel-bin/libci_test
```

## ```Building with CMake```


Currently not hooked up to gtest.

``` bash
mkdir build && cd build && cmake .. && make
```

## ```pybind11```

Currently only two functions are exposed via pybind. TODO: wrap other methods which may be useful (e.g. GFMC).