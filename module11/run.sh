#!/bin/bash

# Compile OpenCL example
g++ Convolution.cpp -o Convolution -framework OpenCL

# Run Convolution using a variety of input signal sizes and masks

# Execute Convolution on 8x8 signal using 3x3 mask from example
./Convolution 0 > convolution_output0.txt

# Execute Convolution on 8x8 signal using 3x3 identity mask
./Convolution 1 > convolution_output1.txt

# Execute Convolution on 49x49 signal using 7x7 mask from homework
./Convolution 2 > convolution_output2.txt

# Execute Convolution on 49x49 signal using 7x7 identity mask
./Convolution 3 > convolution_output3.txt
