#!/bin/sh
# 姓名：丁黄炎
set -eu
cd "$(dirname "$0")"
temporary="libcamera_threads.so.$$"
trap 'rm -f -- "$temporary"' EXIT
g++ -std=c++17 -O3 -fPIC -shared -Wall -Wextra camera_threads.cpp \
    -o "$temporary" $(pkg-config --cflags --libs opencv4) -lEGL -lGLESv2 -lX11 -lXext
mv -f -- "$temporary" libcamera_threads.so
