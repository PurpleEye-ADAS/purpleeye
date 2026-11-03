#!/bin/sh
set -eu
cd "$(dirname "$0")"
temporary="libtraffic_capture.so.$$"
trap 'rm -f -- "$temporary"' EXIT
g++ -std=c++17 -O3 -fPIC -shared -Wall -Wextra -ffp-contract=off \
    pcie_capture.cpp -o "$temporary" $(pkg-config --cflags --libs opencv4)
mv -f -- "$temporary" libtraffic_capture.so
