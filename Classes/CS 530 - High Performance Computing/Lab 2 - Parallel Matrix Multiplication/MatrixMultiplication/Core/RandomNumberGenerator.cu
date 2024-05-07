//
// Created by Strut on 5/6/2024.
//
#include "RandomNumberGenerator.cuh"
#include <iostream>

//defines for error checking and handling
#define CUDA_CALL(x) do { cudaError_t err = (x); if(err != cudaSuccess){ \
    printf("Error %d at: %s:%d\n\t%s\n", err, __FILE__, __LINE__, cudaGetErrorString(err)); \
    exit(EXIT_FAILURE);}} while(0)

#define CURAND_CALL(x) do { curandStatus_t err = (x); if(err !=CURAND_STATUS_SUCCESS) { \
    printf("Error %d at %s:%d\n",err, __FILE__,__LINE__);\
    exit(EXIT_FAILURE);}} while(0)

/** \brief Builds a CuRAND generator from some given initial parameters.
 * Useful for configuring random number generators neatly
 * @param out_gen An out parameter that will be the configured generator upon success
 * @param rng_type The RNG type desired for this generator
 * @param seed An integer that seeds the given generator. Default = 0 means a seed will be randomly generated
 * @param offset An offset used to skip portions of the RNG cycle, effectively a random starting point. Default = 0 means an offset will be randomly generated
 */


__host__ void
build_curand_generator(curandGenerator_t &out_gen, curandRngType_t rng_type, int seed, unsigned int offset)
{
    CURAND_CALL(curandCreateGenerator(&out_gen, rng_type));

    if(seed == 0) seed = static_cast<int>(time(nullptr));
    CURAND_CALL(curandSetPseudoRandomGeneratorSeed(out_gen, seed));

    curandOrdering_t order;
    switch(rng_type)
    {
        case CURAND_RNG_PSEUDO_XORWOW:
            order = CURAND_ORDERING_PSEUDO_SEEDED;
            break;
        case CURAND_RNG_PSEUDO_MRG32K3A:
            order = CURAND_ORDERING_PSEUDO_DYNAMIC;
            break;
        case CURAND_RNG_PSEUDO_MTGP32:
            //order = CURAND_ORDERING_PSEUDO_BEST;
            //break;
        case CURAND_RNG_PSEUDO_MT19937:
            order = CURAND_ORDERING_PSEUDO_BEST;
            break;
        case CURAND_RNG_PSEUDO_PHILOX4_32_10:
            order = CURAND_ORDERING_PSEUDO_DYNAMIC;
            break;
        default:
            order = CURAND_ORDERING_PSEUDO_DEFAULT;
            break;
    }
    CURAND_CALL(curandSetGeneratorOrdering(out_gen, order));

    if(rng_type != CURAND_RNG_PSEUDO_MTGP32 && rng_type != CURAND_RNG_PSEUDO_MT19937) {
        if (offset == 0) {
            unsigned int *random_offset;
            CUDA_CALL(cudaMallocManaged(&random_offset, sizeof(unsigned int)));
            CURAND_CALL(curandGenerate(out_gen, random_offset, 1));
            CUDA_CALL(cudaDeviceSynchronize());
            offset = *random_offset;
            CUDA_CALL(cudaFree(random_offset));
        }

        CURAND_CALL(curandSetGeneratorOffset(out_gen, offset));
    }
}

__host__ void generate_random_numbers(curandGenerator_t generator, long double *outputPtr, size_t num, float lower, float upper)
{
    float *dev_numbers, *host_numbers;
    host_numbers = (float*)(malloc(sizeof(float) * num));
    CUDA_CALL(cudaMalloc(&dev_numbers, sizeof(float) * num));
    CURAND_CALL(curandGenerateUniform(generator, dev_numbers, num));
    CUDA_CALL(cudaDeviceSynchronize());
    cudaMemcpy(host_numbers, dev_numbers, sizeof(float) * num, cudaMemcpyDeviceToHost);
    for(int i = 0; i < num; i++)
    {
        auto n = static_cast<long double>(host_numbers[i]);
        outputPtr[i] = lower + (n * (upper - lower));
    }
    CUDA_CALL(cudaFree(dev_numbers));
    free(host_numbers);
}

