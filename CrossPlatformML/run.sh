#!/bin/bash

# Compile application
g++ -std=c++11 kmeans.cpp gpu_compatibility.cpp -o gpu_compatibility

# Run Application
./gpu_compatibility
