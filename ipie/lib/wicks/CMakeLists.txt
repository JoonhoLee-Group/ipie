cmake_minimum_required(VERSION 3.4...3.18)
project(wicks)

set(CXXFLAGS "-O3 -std=c++11")
set(CMAKE_CXX_FLAGS "${CXXFLAGS}")

#target_compile_definitions(lib_wicks PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
add_library(
    wicks
    SHARED
    ${PROJECT_SOURCE_DIR}/src/bitstring.cc
    ${PROJECT_SOURCE_DIR}/src/excitations.cc
    ${PROJECT_SOURCE_DIR}/src/ci_wavefunction.cc
    )

set_target_properties(
    wicks
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    )
