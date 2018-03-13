#!/bin/bash

nvcc module6.cu -o module6
# For all tests, the blockSize is set to 512

# Test 1: inputSize = 1*blockSize
./module6 1 > output1.txt

# Test 2: inputSize = 10*blockSize
./module6 10 > output2.txt

# Test 3: inputSize = 100*blockSize
./module6 100 > output3.txt

# Test 4: inputSize = 1000*blockSize
./module6 1000 > output4.txt

# Test 5: inputSize = 10000*blockSize
./module6 10000 > output5.txt
