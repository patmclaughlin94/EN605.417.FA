#!/bin/bash

nvcc module5.cu -o module5
# For all tests, the blockSize is set to 512

# Test 1: inputSize = 1*blockSize
./module5 1 > output1.txt

# Test 2: inputSize = 10*blockSize
./module5 10 > output2.txt

# Test 3: inputSize = 100*blockSize
./module5 100 > output3.txt

# Test 4: inputSize = 1000*blockSize
./module5 1000 > output4.txt

# Test 5: inputSize = 10000*blockSize
./module5 10000 > output5.txt
