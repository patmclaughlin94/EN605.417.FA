#!/bin/bash

nvcc module8.cu -lcublas -lcurand -o module8

# Test 1: blockSize = 256, inputSize = 5x5
./module8 256 5 > output1.txt

# Test 2: blockSize = 256, inputSize = 50x50
./module8 256 50 > output2.txt

# Test 3: blockSize = 256, inputSize = 250x250
./module8 256 250 > output3.txt

# Test 4: blockSize = 256, inputSize = 500x500
./module8 256 500 > output4.txt

# Test 5: blockSize = 512, inputSize = 5x5
./module8 512 5 > output5.txt

# Test 6: blockSize = 512, inputSize = 50x50
./module8 512 50 > output6.txt

# Test 7: blockSize = 512, inputSize = 250x250
./module8 512 250 > output7.txt

# Test 8: blockSize = 512, inputSize = 500x500
./module8 512 500 > output8.txt

# Test 9: blockSize = 1024, inputSize = 5x5
./module8 1024 5 > output9.txt

# Test 10: blockSize = 1024, inputSize = 50x50
./module8 1024 50 > output10.txt

# Test 11: blockSize = 1024, inputSize = 250x250
./module8 1024 250 > output11.txt

# Test 12: blockSize = 1024, inputSize = 500x500
./module8 1024 500 > output12.txt

