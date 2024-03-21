#!/bin/bash

bazel build :libci
bazel build :libci.so
bazel build :libci_test

if [[ ! -f libci.so && -f bazel-bin/libci.so ]]; then
    echo "libci.so not found, symlinking"
    ln -s bazel-bin/libci.so libci.so
else
    echo "libci.so already exists"
fi