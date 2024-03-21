cmake_minimum_required(VERSION 3.4...3.18)
project(ci)

set(CXXFLAGS "-O3 -std=c++17 -march=native -flto -Wall -Wextra -Wpedantic")
set(CMAKE_CXX_FLAGS "${CXXFLAGS}")


#target_compile_definitions(lib_wicks PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
file(GLOB_RECURSE TEST_FILES
  "${PROJECT_SOURCE_DIR}/src/*.test.cc"
)
file(GLOB_RECURSE PYBIND_FILES 
  "${PROJECT_SOURCE_DIR}/src/*.pybind.cc"
)
file(GLOB_RECURSE SOURCE_FILES 
  "${PROJECT_SOURCE_DIR}/src/*.cc"
)
list(REMOVE_ITEM SOURCE_FILES ${TEST_FILES})
list(REMOVE_ITEM SOURCE_FILES ${PYBIND_FILES})
add_library(
    ci
    SHARED 
    ${SOURCE_FILES}
    )

set_target_properties(
    ci
    PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}
    )

list(REMOVE_ITEM SOURCE_FILES_NO_MAIN ${TEST_FILES})
list(REMOVE_ITEM SOURCE_FILES_NO_MAIN ${PYBIND_FILES})
find_package(pybind11 CONFIG)
if (${pybind11_FOUND})
  message("this")
  pybind11_add_module(libci_python_bindings ${PYBIND_FILES} ${SOURCE_FILES})
  set_target_properties(libci_python_bindings PROPERTIES OUTPUT_NAME libci)
  set_target_properties(libci_python_bindings PROPERTIES SUFFIX ".so")
  set_target_properties(libci_python_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
endif()

find_package(GTest QUIET)
if(${GTest_FOUND})
  add_executable(libci_test ${SOURCE_FILES} ${TEST_FILES})
  target_link_libraries(libci_test GTest::gtest GTest::gtest_main)
  target_compile_options(libci_test PRIVATE -Wall -Wpedantic -g -fno-omit-frame-pointer -fno-strict-aliasing -fsanitize=undefined -fsanitize=address ${MACHINE_FLAG})
  target_link_options(libci_test PRIVATE -g -fno-omit-frame-pointer -fsanitize=undefined -fsanitize=address)
endif()