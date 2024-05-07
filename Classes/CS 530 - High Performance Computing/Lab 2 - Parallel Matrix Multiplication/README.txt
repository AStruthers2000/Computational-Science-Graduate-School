<<<<<<< HEAD
The main.c file can be compiled using GCC. 

There are four possible command line arguments:
./Matrix <dimension> <power> <matrix_type> <verbosity>

<dimension>: should be an integer value in the range [1, 1000]. This argument is not optional.

<power>: should be an integer value in the range [1, 10000]. This argument is optional, and has a default value of 10.

<matrix_type>: can either be 0 for a randomly initialized matrix, or 1 for an identity matrix. This argument is optional and has a default value of Random.

<verbosity>: the level of output you want with execution. 0 indicates minimal output, and 1 indicates full output. Default value is 0.

If no command line arguments are provided, the default behavior will be to run automated tests. This experimentation is very time consuming and should only be utilized if absolutely necessary.

Example invocation: 
./Matrix 10 92 0 1 - this invocation raises a 10x10 matrix to the 92nd power, initialized with random values with full printing output.

./Matrix bla bla bla bla - this invocation will result in a matrix dimension of 0x0, which will cause a friendly print out message to be output.

=======
The main.c file can be compiled using GCC. 

There are four possible command line arguments:
./Matrix <dimension> <power> <matrix_type> <verbosity>

<dimension>: should be an integer value in the range [1, 1000]. This argument is not optional.

<power>: should be an integer value in the range [1, 10000]. This argument is optional, and has a default value of 10.

<matrix_type>: can either be 0 for a randomly initialized matrix, or 1 for an identity matrix. This argument is optional and has a default value of Random.

<verbosity>: the level of output you want with execution. 0 indicates minimal output, and 1 indicates full output. Default value is 0.

If no command line arguments are provided, the default behavior will be to run automated tests. This experimentation is very time consuming and should only be utilized if absolutely necessary.

Example invocation: 
./Matrix 10 92 0 1 - this invocation raises a 10x10 matrix to the 92nd power, initialized with random values with full printing output.

./Matrix bla bla bla bla - this invocation will result in a matrix dimension of 0x0, which will cause a friendly print out message to be output.

>>>>>>> ba57376202a1955c6a871d7cabfc4da3442fccce
./Matrix 9 - this invocation raises a 9 by 9 matrix to the default power of 10, initialized with random values and minimal information printed to the screen.