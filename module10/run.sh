#!/bin/bash

# Compile OpenCL example
g++ HelloWorld.cpp -o HelloWorld -framework OpenCL

# Run all simple math operations with different size arrays

# Simple Math ops with size 100 Array
./HelloWorld 100 > hello_world_output1.txt

# Simple Math ops with size 1000 Array
./HelloWorld 1000 > hello_world_output2.txt

# Simple Math ops with size 10000 Array
./HelloWorld 10000 > hello_world_output3.txt

# Simple Math ops with size 100000 Array
./HelloWorld 100000 > hello_world_output4.txt
