#!/bin/bash

# https://github.com/pytorch/pytorch#install-pytorch
wget https://download.pytorch.org/libtorch/cu101/libtorch-cxx11-abi-shared-with-deps-1.3.1.zip
tar -xvzf libtorch-cxx11-abi-shared-with-deps-1.3.1.zip libtorch

wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
mkdir nlohmann
mv json.hpp nlohmann