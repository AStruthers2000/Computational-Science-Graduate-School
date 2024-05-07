//
// Created by Strut on 5/6/2024.
//

#include <iostream>
#include "MatrixMultiplier.cuh"



__global__ void
Multiply_ElementWise(const long double *left_matrix, const long double *right_matrix, long double *result, int matrix_size)
{
    auto row = blockIdx.x;
    auto col = threadIdx.x;

    double temp = 0.0;
    for(int i = 0; i < matrix_size; i++)
    {
        auto left_ele = static_cast<double>(left_matrix[row * matrix_size + i]);
        auto right_ele = static_cast<double>(right_matrix[i * matrix_size + col]);
        temp += left_ele * right_ele;
    }

    result[row * matrix_size + col] = temp;
}

__host__ void raise_to_power(const long double *base, long double *previous, long double *current, const int& dimension, const int& power, const MatrixMultiplicationType& type)
{
    for(int i = 0; i < power; i++)
    {
        //first, we want to copy current to previous and clear current
        //however, remember that dev arrays are 1D flattened, so we have to do pointer arithmetic
        for(int row = 0; row < dimension; row++)
        {
            cudaMemcpy(previous + row * dimension, current + row * dimension, sizeof(long double) * dimension, cudaMemcpyDeviceToDevice);
            cudaMemset(current + row * dimension, 0, sizeof(long double) * dimension);
        }

        //now we want to multiply previous by base to calculate current
        switch(type)
        {
            case ElementWise:
                Multiply_ElementWise<<<dimension, dimension>>>(base, previous, current, dimension);
                break;
            case RowWise:
                break;
            case ColWise:
                break;
        }
        cudaDeviceSynchronize();
    }
}


__global__ void PrintMatrix(const long double *matrix, int matrix_size)
{
    for(int i = 0; i < matrix_size; i++)
    {
        for(int j = 0; j < matrix_size; j++)
        {
            printf("%2.4f\t", static_cast<float>(matrix[i * matrix_size + j]));
        }
        printf("\n");
    }
}
