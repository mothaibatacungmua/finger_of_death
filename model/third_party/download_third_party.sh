#!/bin/bash

wget https://download.pytorch.org/libtorch/cu101/libtorch-shared-with-deps-1.3.0.zip
tar -xvzf libtorch-shared-with-deps-1.3.0.zip libtorch

wget https://raw.githubusercontent.com/nlohmann/json/develop/single_include/nlohmann/json.hpp
mkdir nlohmann
mv json.hpp nlohmann