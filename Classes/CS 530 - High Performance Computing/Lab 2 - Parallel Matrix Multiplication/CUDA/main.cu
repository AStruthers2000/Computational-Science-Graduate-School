#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#include "Core/RandomNumberGenerator.cuh"
#include "Core/MatrixMultiplier.cuh"

using namespace std::chrono;

//global memory
long double **host_base, **host_previous, **host_current;
long double *dev_base, *dev_previous, *dev_current;

long double *initial_random_numbers;

void run_experiment(const int& dimension, const int& power);

//helper functions - memory management
void allocate_memory(const int& matrix_dimension);
void initialize_memory(const int& matrix_dimension, curandGenerator_t generator);
void free_memory(const int& matrix_dimension);

//helper functions - random
void print_matrix(long double** matrix, const int& matrix_dimension, const char* msg);

int main()
{
    int num_dimensions = 19;
    int num_powers = 19;
    int dimensions[19] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};
    int powers[19] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000};

    for(int i = 0; i < num_dimensions; i++)
    {
        run_experiment(dimensions[i], 1000);
    }
    printf("\n=============================================================\n\n");
    for(int i = 0; i < num_powers; i++)
    {
        run_experiment(50, powers[i]);
    }
}


void run_experiment(const int &dimension, const int &power)
{
    for(int i = 0; i < 30; i++)
    {
        curandGenerator_t prng;
        build_curand_generator(prng, CURAND_RNG_PSEUDO_XORWOW, 1);

        allocate_memory(dimension);
        initialize_memory(dimension, prng);

        /*
        print_matrix(host_base, matrix_dimension, "Host matrix");
        print_matrix(host_previous, matrix_dimension, "Host previous");
        print_matrix(host_current, matrix_dimension, "Host current");

        printf("Dev base:\n");
        PrintMatrix<<<1, 1>>>(dev_base, matrix_dimension);
        cudaDeviceSynchronize();

        printf("Dev previous:\n");
        PrintMatrix<<<1, 1>>>(dev_previous, matrix_dimension);
        cudaDeviceSynchronize();

        printf("Dev current:\n");
        PrintMatrix<<<1, 1>>>(dev_current, matrix_dimension);
        cudaDeviceSynchronize();
         */
        auto start = high_resolution_clock::now();

        raise_to_power(
                dev_base,
                dev_previous,
                dev_current,
                dimension,
                power,
                ElementWise
        );

        auto stop = high_resolution_clock::now();
        auto duration = static_cast<double>(duration_cast<microseconds>(stop - start).count()) / 1e06;

        printf("%d,%d,%.6f\n", dimension, power, duration);

        /*
        printf("Dev base:\n");
        PrintMatrix<<<1, 1>>>(dev_base, matrix_dimension);
        cudaDeviceSynchronize();

        printf("Dev previous:\n");
        PrintMatrix<<<1, 1>>>(dev_previous, matrix_dimension);
        cudaDeviceSynchronize();

        printf("Dev current:\n");
        PrintMatrix<<<1, 1>>>(dev_current, matrix_dimension);
        cudaDeviceSynchronize();
         */

        free_memory(dimension);
    }
}




void allocate_memory(const int& matrix_dimension)
{
    //host allocation
    host_base = (long double**)(malloc(sizeof(long double*) * matrix_dimension));
    host_previous = (long double**)(malloc(sizeof(long double*) * matrix_dimension));
    host_current = (long double**)(malloc(sizeof(long double*) * matrix_dimension));

    for(int i = 0; i < matrix_dimension; i++)
    {
        host_base[i] = (long double*)(malloc(sizeof(long double) * matrix_dimension));
        host_previous[i] = (long double*)(malloc(sizeof(long double) * matrix_dimension));
        host_current[i] = (long double*)(malloc(sizeof(long double) * matrix_dimension));

        memset(host_base[i], 0, sizeof(long double) * matrix_dimension);
        memset(host_previous[i], 0, sizeof(long double) * matrix_dimension);
        memset(host_current[i], 0, sizeof(long double) * matrix_dimension);
    }

    //cuda allocation
    cudaMalloc(&dev_base, sizeof(long double*) * matrix_dimension * matrix_dimension);
    cudaMalloc(&dev_previous, sizeof(long double*) * matrix_dimension * matrix_dimension);
    cudaMalloc(&dev_current, sizeof(long double*) * matrix_dimension * matrix_dimension);

    //other allocation
    initial_random_numbers = (long double*)(malloc(sizeof(long double) * matrix_dimension * matrix_dimension));

}

void initialize_memory(const int& matrix_dimension, curandGenerator_t generator)
{
    //generate matrix_dimension^2 random numbers
    generate_random_numbers(generator, initial_random_numbers, matrix_dimension * matrix_dimension, -1.0, 1.0);

    //initialize host matrices
    for(int i = 0; i < matrix_dimension; i++)
    {
        memcpy(host_base[i], initial_random_numbers + i * matrix_dimension, sizeof(long double) * matrix_dimension);
        memcpy(host_previous[i], initial_random_numbers + i * matrix_dimension, sizeof(long double) * matrix_dimension);
        memcpy(host_current[i], initial_random_numbers + i * matrix_dimension, sizeof(long double) * matrix_dimension);
    }

    //copy 2D host matrices to 1D device matrices
    for(int i = 0; i < matrix_dimension; i++)
    {
        cudaMemcpy(dev_base + i * matrix_dimension, host_base[i], sizeof(long double) * matrix_dimension, cudaMemcpyHostToDevice);
    }
    cudaMemcpy(dev_previous, dev_base, sizeof(long double) * matrix_dimension * matrix_dimension, cudaMemcpyDeviceToDevice);
    cudaMemcpy(dev_current, dev_base, sizeof(long double) * matrix_dimension * matrix_dimension, cudaMemcpyDeviceToDevice);
}

void free_memory(const int& matrix_dimension)
{
    for(int i = 0; i < matrix_dimension; i++)
    {
        free(host_base[i]);
        free(host_previous[i]);
        free(host_current[i]);
    }

    free(host_base);
    free(host_previous);
    free(host_current);

    cudaFree(dev_base);
    cudaFree(dev_previous);
    cudaFree(dev_current);
}

void print_matrix(long double **matrix, const int &matrix_dimension, const char *msg)
{
    printf("%s:\n", msg);
    for(int i = 0; i < matrix_dimension; i++)
    {
        for(int j = 0; j < matrix_dimension; j++)
        {
            printf("%2.4Lf\t", matrix[i][j]);
        }
        printf("\n");
    }

}

