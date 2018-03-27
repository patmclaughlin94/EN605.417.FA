#!/bin/bash

nvcc stream_example.cu -o stream_example

# Test 1: blockSize = 256, inputSize = 64x64
./stream_example 256 4096 > output1.txt

# Test 2: blockSize = 256, inputSize = 256x256
./stream_example 256 65536 > output2.txt

# Test 3: blockSize = 256, inputSize = 512x512
./stream_example 256 262144 > output3.txt

# Test 4: blockSize = 256, inputSize = 1024x1024
./stream_example 256 1048576 > output4.txt

# Test 5: blockSize = 512, inputSize = 64x64
./stream_example 512 4096 > output5.txt

# Test 6: blockSize = 512, inputSize = 256x256
./stream_example 512 65536 > output6.txt

# Test 7: blockSize = 512, inputSize = 512x512
./stream_example 512 262144 > output7.txt

# Test 8: blockSize = 512, inputSize = 1024x1024
./stream_example 512 1048576 > output8.txt

# Test 9: blockSize = 1024, inputSize = 64x64
./stream_example 1024 4096 > output9.txt

# Test 10: blockSize = 1024, inputSize = 256x256
./stream_example 1024 65536 > output10.txt

# Test 11: blockSize = 1024, inputSize = 512x512
./stream_example 1024 262144 > output11.txt

# Test 12: blockSize = 1024, inputSize = 1024x1024
./stream_example 1024 1048576 > output12.txt

