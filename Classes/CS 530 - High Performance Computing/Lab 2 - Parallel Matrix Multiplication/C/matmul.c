#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <float.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>


#define MATRIX_TYPE long double

#define TEST_DIMENSION 200
#define TEST_POWER 10

#define DEBUG 0

typedef enum
{
	Identity = 0,
	Random = 1
} MatrixInitType;

typedef enum
{
	ElementWise,
	RowWise,
	ColWise
} MatrixMultType;

typedef enum
{
	Error,
	Standard,
	FullTest
} OperationType;

MATRIX_TYPE** matrix;
MATRIX_TYPE** previous;
MATRIX_TYPE** current;


void initialize_matrix(MatrixInitType init_type, int dimension);
void run_experiment(MatrixInitType init_type, int matrix_dimension, int power, MatrixMultType wise_type);

MATRIX_TYPE generate_random();
void allocate_memory(int dimension);
void free_memory(int dimension);


int main(int argc, char** argv)
{
	//parse args
	int dimension = TEST_DIMENSION;
	int power = TEST_POWER;
	OperationType test_type = Error;
	MatrixInitType matrix_init_type = Random;
	MatrixMultType matrix_wise_type = RowWise;
	if(argc >= 3)
	{
		dimension = atoi(argv[1]);
		power = atoi(argv[2]);
		test_type = Standard;
		
		if(argc >= 4)
		{
			matrix_init_type = atoi(argv[3]) == 0 ? Random : Identity;
		}
		if(argc >= 5)
		{
			int type = atoi(argv[4]);
			matrix_wise_type = type == 0 ? RowWise : type == 1 ? ColWise : ElementWise;
		}
		
		
		if(dimension < 1 || dimension > 1000)
		{
		        printf("Matrix dimension outside of acceptable range. Please enter a number in the range: [1, 1000]\n");
		        return 1;
		}
		if(power < 0 || power > 10000)
		{
		        printf("Matrix power outside of acceptable range. Please enter a number in the range: [0, 10000]\n");
		        return 2;
		}
	}
	else if(argc == 1)
	{
		test_type = FullTest;
	}
	
	if(test_type == Error)
	{
		printf("Some error occurred while parsing args, exiting\n");
		return 3;
	}
	
	//allocate memory
	
	switch(test_type)
	{
	case Standard:
		printf("Starting standard matrix multiplication with parameters:\n\tMatrix dimension:\t%d\n\tMatrix power:\t\t%d\n\tInitialization:\t\t%s\n\tMultiplication style:\t%s\n", dimension, power, matrix_init_type == 0 ? "Identity" : "Random", matrix_wise_type == RowWise ? "Row wise" : matrix_wise_type == ColWise ? "Column wise" : "Element wise");
		
		run_experiment(matrix_init_type, dimension, power, matrix_wise_type);
		break;
		
	case FullTest:
		printf("Currently not implemented\n");
		break;
		
	default: break;
	}
	
}








void run_experiment(MatrixInitType init_type, int matrix_dimension, int power,  MatrixMultType wise_type)
{
        for(int i = 0; i < 30; i++)
        {
                allocate_memory(matrix_dimension);

                srand(time(NULL));
                initialize_matrix(init_type, matrix_dimension);

                struct timeval timeofday_start, timeofday_end;
                double timeofday_elapsed;

                gettimeofday(&timeofday_start, NULL);

                //raise_to_power(matrix_dimension, power, 0);
                //actually run experiment

                gettimeofday(&timeofday_end, NULL);

                timeofday_elapsed = (timeofday_end.tv_sec -timeofday_start.tv_sec) + (timeofday_end.tv_usec - timeofday_start.tv_usec) / 1000000.0;

                printf("%d,%d,%.6f\n", matrix_dimension, power, timeofday_elapsed);

                free_memory(matrix_dimension);
        }
}














/**
* Initializes matrix with either the identity matrix or random values
*/
void initialize_matrix(MatrixInitType init_type, int dimension)
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

