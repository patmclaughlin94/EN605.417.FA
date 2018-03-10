#!/bin/bash

nvcc dynamicDataSize.cu -L /usr/local/cuda/lib -lcudart -o dynamicDataSize

# NOTE: Throughout these tests, the array size is kept constant at 1024

# Test 1: 1024 Threads, 32 Threads / Block
./dynamicDataSize 1024 32 > output1.txt
#./dynamicDataSize < 1024 32 > output1.txt
# Test 2: 1024 Threads, 64 Threads / Block
./dynamicDataSize 1024 64 > output2.txt

# Test 3: 1024 Threads, 50 Threads / Block
./dynamicDataSize 1024 50 > output3.txt

# Test 4: 512 Threads, 32 Threads / Block
./dynamicDataSize 512 32 > output4.txt

