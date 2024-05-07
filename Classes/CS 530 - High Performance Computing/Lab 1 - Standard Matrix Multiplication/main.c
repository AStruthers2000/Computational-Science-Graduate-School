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
int read_command_args(int argc, char** argv, int* out_dimension, int* out_power, MatrixInitialization* out_type, int* out_verbose);
long double generate_random();
void print_matrix(long double** m, int dimension);
void allocate_memory(int dimension);
void free_memory(int dimension);


/***** Function implementations *****/

/**
* Main function
*/
int main(int argc, char** argv)
{
	if(argc > 1)
	{
		int matrix_dimension = 0;
		int power = 0;
		MatrixInitialization type;
		int verbose = 0;

		int err = read_command_args(argc, argv, &matrix_dimension, &power, &type, &verbose);
		if(err)
		{
			printf("Error while reading command line args, see log. Exiting with code: %d\n", err);
			return err;
		}




		allocate_memory(matrix_dimension);

		srand(time(NULL));
		initialize_matrix(type, matrix_dimension);

		if(DEBUG)
		{
			printf("\nBase matrix:\n");
			print_matrix(matrix, matrix_dimension);

			printf("\nPrevious matrix:\n");
			print_matrix(previous, matrix_dimension);

			printf("\nCurrent matrix:\n");
			print_matrix(current, matrix_dimension);
		}

		printf("Base matrix:\n");
		print_matrix(matrix, matrix_dimension);

		printf("\n========== Starting matrix multiplication ==========\n");

		clock_t clock_start, clock_end;
		double clock_elapsed;

		struct timeval timeofday_start, timeofday_end;
		double timeofday_elapsed;

		time_t time_start, time_end;
		double time_elapsed;

		clock_start = clock();
		gettimeofday(&timeofday_start, NULL);
		time_start = time(NULL);

		raise_to_power(matrix_dimension, power, verbose);

		clock_end = clock();
		gettimeofday(&timeofday_end, NULL);
		time_end = time(NULL);

		printf("\n========== Finishing matrix multiplication ==========\n");

		printf("\nFinal matrix:\n");
		print_matrix(current, matrix_dimension);

		clock_elapsed = ((double)(clock_end - clock_start)) / CLOCKS_PER_SEC;
		timeofday_elapsed = (timeofday_end.tv_sec -timeofday_start.tv_sec) + (timeofday_end.tv_usec - timeofday_start.tv_usec) / 1000000.0;
		time_elapsed = difftime(time_end, time_start);

		printf("Elapsed time using clock():        \t%.6f seconds\n", clock_elapsed);
		printf("Elapsed time using gettimeofday(): \t%.6f seconds\n", timeofday_elapsed);
		printf("Elapsed time using time():         \t%.6f seconds\n", time_elapsed);

		//printf("%d,%d,%.6f\n", matrix_dimension, power, timeofday_elapsed);

		free_memory(matrix_dimension);
	}
	else
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

		printf("%d,%d,%.6f\n", matrix_dimension, power, timeofday_elapsed);

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

		if(verbose) printf("Raising matrix to the power of %d\n", i);

		//now we want to multiply previous by base, to calculate current
		int v = multiply_matrix(dimension);

		if(DEBUG || v)
		{
			if(verbose)
			{
				printf("\nBase matrix:\n");
				print_matrix(matrix, dimension);

				printf("\nPrevious matrix:\n");
				print_matrix(previous, dimension);

				printf("\nCurrent matrix:\n");
				print_matrix(current, dimension);

				printf("\n\n\n--------------------\n\n\n");
			}

			if(v) break;
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
* Read command line arguments and generate errors/warnings when unexpected args are passed
*/
int read_command_args(int argc, char** argv, int* out_dimension, int* out_power, MatrixInitialization* out_type, int* out_verbose)
{
	//default values, used if no arguments are provided
	*out_dimension = TEST_DIMENSION;
	*out_power = TEST_POWER;
	*out_type = Random;
	*out_verbose = 0;

	/** initialize command line arguments for dimensionality and power **/

	if(argc == 2)
	{
		//if only one argument is provided, we will assume its the dimension
		*out_dimension = atoi(argv[1]);
		printf("Warning: only one argument was provided, assumption is that this is the dimension of the matrix\n");
	}

	if(argc >= 3)
	{
		*out_dimension = atoi(argv[1]);
		*out_power = atoi(argv[2]);

		printf("Provided dimension: \t%d\nProvided power: \t%d\n", *out_dimension, *out_power);
	}

	if(argc >= 4)
	{
		*out_type = atoi(argv[3]) == 0 ? Random : Identity;
		printf("Provided type:  \t%s\n", *out_type == 0 ? "Identity" : "Random");
	}

	if(argc >= 5)
	{
		*out_verbose = atoi(argv[4]) == 1 ? 1 : 0;
		printf("Provided verbose: \t%s\n", *out_verbose == 0 ? "Quiet" : "Detailed");
	}
	printf("\n");

	if(argc > 5)
	{
		printf("Warning: too many command line arguments provided, only using recognized values\n");
	}

	if(*out_dimension < 1 || *out_dimension > 1000)
	{
		printf("Matrix dimension outside of acceptable range. Please enter a number in the range: [1, 1000]\n");
		return 1;
	}
	if(*out_power < 0 || *out_power > 10000)
	{
		printf("Matrix power outside of acceptable range. Please enter a number in the range: [0, 10000]\n");
		return 2;
	}

	return 0;
}

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
* Print matrix
*/
void print_matrix(long double** m, int dimension)
{
	for(int i = 0; i < dimension; i++)
	{
		for(int j = 0; j < dimension; j++)
		{
			if(m[i][j] < 0 && m[i][j] > -10000) 	 printf("%.4Lf\t", m[i][j]);
			else if(m[i][j] >= 0 && m[i][j] < 10000) printf("%.5Lf\t", m[i][j]);
			else if(m[i][j] >= 10000)		 printf("%.5Le\t", m[i][j]);
			else 					 printf("%.4Le\t", m[i][j]);
		}
		printf("\n");
	}
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
