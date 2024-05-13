#include "matthread.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

void* multiply_row(void* arg)
{
	thread_data* data = (thread_data*) arg;

	long double** base = data->base;
	long double** prev = data->prev;
	long double** result = data->result;

	int dimension = data->dimension;
	int row = data->index;
	int col = 0, k = 0;


	//memset(result[row], 0, sizeof(long double) * dimension);
	for(col = 0; col < dimension; col++)
	{
		//result[row][col] = 0;
		for(k = 0; k < dimension; k++)
		{
			result[row][col] += base[row][k] * prev[k][col];
		}
	}

	pthread_exit(NULL);
}

void* multiply_col(void* arg)
{
	thread_data* data = (thread_data*) arg;

	long double** base = data->base;
	long double** prev = data->prev;
	long double** result = data->result;

	int dimension = data->dimension;
	int col = data->index;
	int row = 0, k = 0;


	//memset(result[row], 0, sizeof(long double) * dimension);
	for(row = 0; row < dimension; row++)
	{
		//result[row][col] = 0;
		for(k = 0; k < dimension; k++)
		{
			result[row][col] += base[row][k] * prev[k][col];
		}
	}

	pthread_exit(NULL);
}

void* multiply_element(void* arg)
{
	thread_data* data = (thread_data*) arg;

	long double** base = data->base;
	long double** prev = data->prev;
	long double** result = data->result;

	int dimension = data->dimension;
	int row = data->index / dimension;
	int col = data->index % dimension; 
	int k = 0;


	//memset(result[row], 0, sizeof(long double) * dimension);
	//result[row][col] = 0;

	for(k = 0; k < dimension; k++)
	{
		result[row][col] += base[row][k] * prev[k][col];
	}

	pthread_exit(NULL);
}
