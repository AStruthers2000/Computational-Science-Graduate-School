//
// Created by Strut on 5/6/2024.
//

#ifndef MATRIXMULTIPLICATION_RANDOMNUMBERGENERATOR_CUH
#define MATRIXMULTIPLICATION_RANDOMNUMBERGENERATOR_CUH

#include <cuda_runtime.h>
#include <curand.h>


__host__ void build_curand_generator(curandGenerator_t &out_gen, curandRngType_t rng_type, int seed = 0, unsigned int offset = 0);

__host__ void generate_random_numbers(curandGenerator_t generator, long double *outputPtr, size_t num, float lower = 0.0, float upper = 1.0);


#endif //MATRIXMULTIPLICATION_RANDOMNUMBERGENERATOR_CUH
