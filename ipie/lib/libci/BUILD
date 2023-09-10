package(default_visibility=["//visibility:public"])

load("@rules_python//python:packaging.bzl", "py_wheel")

SOURCE_FILES = glob(
    [
        "src/*.cc",
        "src/*.h",
    ],
    exclude=glob(["src/*.test.cc", "src/*.pybind.cc", "src/*.pybind.h"]),
)

PYBIND_FILES = glob(
    [
        "src/*.pybind.cc",
        "src/*.pybind.h",
    ],
)

TEST_FILES = glob(
    [
        "src/*.test.cc",
    ],
)


cc_library(
    name="libci",
    srcs=SOURCE_FILES,
    copts=[
        "-std=c++17",
        "-Wall",
        "-Wextra",
        "-pedantic",
        "-O3",
        "-flto",
       "-march=native",
    ],
    includes=["src/"],
)

cc_test(
    name="libci_test",
    srcs=SOURCE_FILES + TEST_FILES,
    copts=[
        "-g",
        "-std=c++17",
        "-march=native",
        "-fsanitize=address",
        "-fsanitize=undefined",
        "-fno-omit-frame-pointer",
        "-fno-strict-aliasing",
        "-Wall",
        "-Wextra",
        "-pedantic",
    ],
    linkopts=[
        "-fsanitize=address",
        "-fsanitize=undefined",
    ],
    includes=["src/"],
    deps=[
        "@gtest",
        "@gtest//:gtest_main",
    ],
)

cc_binary(
    name="libci.so",
    srcs=SOURCE_FILES + PYBIND_FILES,
    copts=[
        "-O3",
        "-std=c++17",
        "-fvisibility=hidden",
        "-march=native",
        "-flto",
    ],
    includes=["src/"],
    linkshared=1,
    deps=["@pybind11"],
)
