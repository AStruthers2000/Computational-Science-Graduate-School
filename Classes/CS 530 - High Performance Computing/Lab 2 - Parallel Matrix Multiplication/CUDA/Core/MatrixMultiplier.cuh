//
// Created by Strut on 5/6/2024.
//

#ifndef MATRIXMULTIPLICATION_MATRIXMULTIPLIER_CUH
#define MATRIXMULTIPLICATION_MATRIXMULTIPLIER_CUH

#include <cuda_runtime.h>

enum MatrixMultiplicationType
{
    ElementWise,
    RowWise,
    ColWise
};

__host__ void raise_to_power(const long double *base, long double *previous, long double *current, const int& dimension, const int& power, const MatrixMultiplicationType& type = ElementWise);
__host__ void multiply_matrix(const int& dimension);
__global__ void Multiply_ElementWise(const long double** left_matrix, const long double** right_matrix, long double** result, int matrix_size);
__global__ void PrintMatrix(const long double* matrix, int matrix_size);


#endif //MATRIXMULTIPLICATION_MATRIXMULTIPLIER_CUH
