#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>

#define TEST_DIMENSION 200
#define TEST_POWER 10

#define DEBUG 0

long double** matrix;
long double** previous;
long double** current;

typedef enum
{
	Identity = 0,
	Random = 1
} MatrixInitialization;

/***** Function signatures *****/

/** Important functions **/
void run_experiment(int matrix_dimension, int power);
void initialize_matrix(MatrixInitialization init_type, int dimension);
void raise_to_power(int dimension, int power, int verbose);
int multiply_matrix(int dimension);

/** Helper functions **/
long double generate_random();
void allocate_memory(int dimension);
void free_memory(int dimension);


/***** Function implementations *****/

/**
* Main function
*/
int main(int argc, char** argv)
{
	int num_dimensions = 14;
	int num_powers = 19;
	int dimensions[14] = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500};
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

	return 0;
}

void run_experiment(int matrix_dimension, int power)
{
	for(int i = 0; i < 30; i++)
	{
		allocate_memory(matrix_dimension);

		srand(time(NULL));
		initialize_matrix(Random, matrix_dimension);

		struct timeval timeofday_start, timeofday_end;
		double timeofday_elapsed;

		gettimeofday(&timeofday_start, NULL);

		raise_to_power(matrix_dimension, power, 0);

		gettimeofday(&timeofday_end, NULL);

		timeofday_elapsed = (timeofday_end.tv_sec -timeofday_start.tv_sec) + (timeofday_end.tv_usec - timeofday_start.tv_usec) / 1000000.0;

		printf("%d,%d,%.6f,Ofast\n", matrix_dimension, power, timeofday_elapsed);

		free_memory(matrix_dimension);
	}
}


/**
* Initializes matrix with either the identity matrix or random values
*/
void initialize_matrix(MatrixInitialization init_type, int dimension)
{
	for(int i = 0; i < dimension; i++)
	{
		for(int j = 0; j < dimension; j++)
		{
			switch(init_type)
			{
			case Identity:
				matrix[i][j] = i == j ? 1 : 0;
				break;
			case Random:
				matrix[i][j] = generate_random();
				break;
			default:
				printf("Error! MatrixInitialization type not defined\n");
				break;
			}
		}
		memcpy(previous[i], matrix[i], sizeof(long double) * dimension);
		memcpy(current[i], matrix[i], sizeof(long double) * dimension);
	}
}


/**
* Raises matrix to a given power by repeatedly invoking multiply_matrix
*/
void raise_to_power(int dimension, int power, int verbose)
{
	for(int i = 0; i < power; i++)
	{
		//first, we want to copy current to previous and clear current
		for(int row = 0; row < dimension; row++)
		{
			memcpy(previous[row], current[row], sizeof(long double) * dimension);
			memset(current[row], 0, sizeof(long double) * dimension);
		}

		//now we want to multiply previous by base, to calculate current
		int v = multiply_matrix(dimension);

		if(v)
		{
			break;
		}
	}
}

int multiply_matrix(int dimension)
{
	int all_nan = 1;
	for(int i = 0; i < dimension; i++)
	{
		for(int j = 0; j < dimension; j++)
		{
			for(int k = 0; k < dimension; k++)
			{
				current[i][j] += previous[i][k] * matrix[k][j];
			}
			if(isnormal(current[i][j]))
			{
				all_nan = 0;
			}
		}
	}
	if(all_nan)
	{
		printf("All entries in matrix are NaN-like\n");
		return 1;
	}
	return 0;
}

/** Helper functions **/

/**
* Generate random long double between [-1.0, 1.0]
*/
long double generate_random()
{
	int rand_int = rand();
	long double norm = (long double)rand_int / RAND_MAX;
	return -1.0 + 2.0 * norm;
}


/**
* Allocate dynamic memory
*/
void allocate_memory(int dimension)
{
	//printf("Allocating a %dx%d matrix\n", dimension, dimension);
	matrix = (long double**)(malloc(sizeof(long double*) * dimension));
	previous = (long double**)(malloc(sizeof(long double*) * dimension));
	current = (long double**)(malloc(sizeof(long double*) * dimension));

	for(int i = 0; i < dimension; i++)
	{
		matrix[i] = (long double*)(malloc(sizeof(long double) * dimension));
		previous[i] = (long double*)(malloc(sizeof(long double) * dimension));
		current[i] = (long double*)(malloc(sizeof(long double) * dimension));
	}

}


/**
* Free dynamic memory
*/
void free_memory(int dimension)
{
	//printf("Freeing a %dx%d matrix \n", dimension, dimension);
	for(int i = 0; i < dimension; i++)
	{
		free(matrix[i]);
		free(previous[i]);
		free(current[i]);
	}

	free(matrix);
	free(previous);
	free(current);
}
