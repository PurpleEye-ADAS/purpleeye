#!/bin/sh
set -eu
cd "$(dirname "$0")"
g++ -std=c++17 -O3 -fPIC -shared -Wall -Wextra \
    pcie_capture.cpp -o libtraffic_capture.so $(pkg-config --cflags --libs opencv4)
