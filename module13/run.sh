# Test Harness
# first command line argument: blocking selection (1 = blocking, 0 = non blocking)
# second command line argument: array size for simple operation arrays
# Simple operations include:
#   - add
#   - sub
#   - mult
#   - div

# Compile Application
g++ events.cpp -o events -framework OpenCL


# BLOCKING KERNEL EXECUTION AND TIMING #
# Will execute "add", "sub", "mult", and "div" in order
# where the output of "add" serves as an input to "sub" and so forth
# Array Size = 100
./events 1 100 > output_1.txt

# Array Size = 1000
./events 1 1000 > output_2.txt

# Array Size = 10000
./events 1 10000 > output_3.txt

# Array Size = 100000
./events 1 100000 > output_4.txt

# NON BLOCKING KERNEL EXECUTION AND TIMING #
# Will execute in arbitrary order
# Array Size = 100
./events 0 100 > output_5.txt

# Array Size = 1000
./events 0 1000 > output_6.txt

# Array Size = 10000
./events 0 10000 > output_7.txt

# Array Size = 100000
./events 0 100000 > output_8.txt