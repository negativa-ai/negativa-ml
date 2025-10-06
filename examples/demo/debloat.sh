#!/bin/bash

# Build the demo
./build.sh


# Use negativa-ml to trace, locate and reconstruct ml libraries
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/extras/CUPTI/lib64

cd build

# trace and locate
negativa_ml debloat -- $PWD/main matmul

# reconstruct
negativa_ml reconstruct --span-path ./nml_workspace/spans/libdemo.so.json    --output-dir ./reconstruct

# replace the original libdemo.so with the reconstructed one, to check if the reconstruction is successful
cp libdemo.so  ./libdemo.so.bak
cp ./reconstruct/libdemo.so  ./libdemo.so

# compare the md5sum of the original and reconstructed libdemo.so
md5sum libdemo.so.bak libdemo.so

# check the exit code of the command
$PWD/main matmul

# restore the original libdemo.so
mv ./libdemo.so.bak libdemo.so
md5sum ./libdemo.so

if [ $? -ne 0 ]; then
    echo "The reconstructed libdemo.so does not work!"
    exit 1
else
    echo "The reconstructed libdemo.so works!"
fi


