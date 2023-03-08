#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// core functions
int* kNN(const float* data, int nData, int k);
float IE_xy(const float* data_x, const float* data_y, int nData_x, int nData_y, int k);
float* unique(const float* data, int nData, int* unique_count);
void sort(int* data, int nData);

// support functions
int xInArray(const float* a, int size, float x);
void print_array(const float* a, int size, const char* msg);
