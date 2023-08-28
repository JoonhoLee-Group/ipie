package(default_visibility=["//visibility:public"])

load("@rules_python//python:packaging.bzl", "py_wheel")

SOURCE_FILES = glob(
    [
        "src/*.cc",
        "src/*.h",
    ],
    exclude=glob(["src/*.test.cc"]),
)

TEST_FILES = glob(
    [
        "src/*.test.cc",
    ],
)


cc_library(
    name="libwicks",
    srcs=SOURCE_FILES,
    copts=[
        "-std=c++17",
    ],
    includes=["src/"],
)

cc_test(
    name="libwicks_test",
    srcs=SOURCE_FILES + TEST_FILES,
    copts=[
        "-std=c++17",
        "-march=native",
        "-Wall",
    ],
    includes=["src/"],
    deps=[
        "@gtest",
        "@gtest//:gtest_main",
    ],
)

cc_binary(
    name="libwicks.so",
    srcs=SOURCE_FILES,
    copts=[
        "-O3",
        "-std=c++17",
        "-fvisibility=hidden",
        "-march=native",
    ],
    includes=["src/"],
    linkshared=1,
    deps=["@gtest"],
)
