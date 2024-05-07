cmake_minimum_required(VERSION 3.27)
project(MatrixMultiplication CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --extended-lambda -g -lineinfo")

add_executable(MatrixMultiplication main.cu
        Core/RandomNumberGenerator.cu
        Core/RandomNumberGenerator.cuh
        Core/MatrixMultiplier.cu
        Core/MatrixMultiplier.cuh)

target_link_libraries(MatrixMultiplication PRIVATE CUDA::curand)

set_target_properties(
        MatrixMultiplication
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
