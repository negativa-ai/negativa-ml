#!/bin/sh
mkdir -p build
nvcc  --gpu-architecture=compute_70 --gpu-code=sm_75,sm_70 --compiler-options '-fPIC' -o ./build/libdemo.so --shared ./demo.cu

g++ -g -O0 main.cpp -L./build -ldemo -o ./build/main -Wl,-rpath,${NEGATIVA_ML_PATH}/examples/demo/build
