#include "pthread_problems.h"

/*
   Jump to line 194 to view Problem Two
   Jump to line 283 to view helper UI and execution functions
   Jump to line 409 to view main function
*/

/* Begin Problem One specific code */
int convertToSequence(int r, int c)
{
	int i = 0;
	i = r * num_elements + c;
	if(i < 0 || i > num_elements * num_elements)
	{
		printf("Index outside of matrix range!!!");
	}
	return i;
}

void printMatrix()
{
	int r, c, i;
	for(r = 0; r < num_elements; r++)
	{
		for(c = 0; c < num_elements; c++)
		{
			i = r * num_elements + c;
			printf("%d", matrix[i]);
		}
		printf("\n");
	}
	printf("\n\n====================\n\n");
}

void* matrixChecker(void* arg)
{
	problem_one_running = 1;

	while(1)
	{
		int r, c, i = 0;
		int sum = 0;
		pthread_mutex_lock(&mutex);
		for(r = 0; r < num_elements; r++)
		{
			for(c = 0; c < num_elements; c++)
			{
				i = r * num_elements + c;
				sum += matrix[i];
			}
		}
		pthread_mutex_unlock(&mutex);

		if(sum == 0 || sum == num_elements*num_elements)
		{
			problem_one_running = 0;
			return (void*) 0;
		}

		sleep(0.1);	//we don't need to check for completion as fast as possible
	}
}

void* matrixThread(void* arg)
{
	int me = (int) arg;
	printf("Started matrix thread %d\n", me);
	while(problem_one_running)
	{
		int i = 0, j = 0;
		i = (rand() % (num_elements));
		j = (rand() % (num_elements));
		int sum = 0, half_sum = 0;
		int r, c;

		pthread_mutex_lock(&mutex);
		int prev_cell = matrix[convertToSequence(i, j)];

		//pesky boudary conditions
		if(i == 0)
		{
			half_sum = 2;
			if(j == 0)
			{
				sum = matrix[convertToSequence(i, j + 1)] +
				      matrix[convertToSequence(i + 1, j)] +
				      matrix[convertToSequence(i + 1, j + 1)];
			}
			else if(j == num_elements - 1)
			{
				sum = matrix[convertToSequence(i, j - 1)] +
				      matrix[convertToSequence(i + 1, j)] +
				      matrix[convertToSequence(i + 1, j - 1)];
			}
			else
			{
				sum = matrix[convertToSequence(i, j - 1)] +
				      matrix[convertToSequence(i + 1, j)] +
				      matrix[convertToSequence(i, j + 1)] +
				      matrix[convertToSequence(i + 1, j - 1)] +
				      matrix[convertToSequence(i + 1, j + 1)];

				half_sum = 3;
			}
		}
		else if(i == num_elements - 1)
		{
			half_sum = 2;
			if(j == 0)
			{
				sum = matrix[convertToSequence(i - 1, j)] +
				      matrix[convertToSequence(i, j + 1)] +
				      matrix[convertToSequence(i - 1, j + 1)];
			}
			else if(j == num_elements - 1)
			{
				sum = matrix[convertToSequence(i, j - 1)] +
				      matrix[convertToSequence(i - 1, j)] +
				      matrix[convertToSequence(i - 1, j - 1)];
			}
			else
			{
				sum = matrix[convertToSequence(i, j - 1)] +
				      matrix[convertToSequence(i, j + 1)] +
				      matrix[convertToSequence(i - 1, j)] +
				      matrix[convertToSequence(i - 1, j - 1)] +
				      matrix[convertToSequence(i - 1, j + 1)];

				half_sum = 3;
			}
		}

		else if(j == 0)
		{
			//the four corners have already been solved by the code that checks the rows
			//so all this needs to do is check the top row
			sum = matrix[convertToSequence(i - 1, j)] +
			      matrix[convertToSequence(i + 1, j)] +
			      matrix[convertToSequence(i, j + 1)] +
			      matrix[convertToSequence(i - 1, j + 1)] +
			      matrix[convertToSequence(i + 1, j + 1)];

			half_sum = 3;
		}
		else if(j == num_elements - 1)
		{
			sum = matrix[convertToSequence(i - 1, j)] +
			      matrix[convertToSequence(i + 1, j)] +
			      matrix[convertToSequence(i, j - 1)] +
			      matrix[convertToSequence(i - 1, j - 1)] +
			      matrix[convertToSequence(i + 1, j - 1)];

			half_sum = 3;
		}

		else
		{
			half_sum = 4;
			for(r = -1; r < 2; r++)
			{
				for(c = -1; c < 2; c++)
				{
					if(!(r == 0 && c == 0)) //we don't want to add the element at (i, j)
					{
						sum += matrix[convertToSequence(i + r, j + c)];
					}
				}
			}
		}

		if(sum < 0 || sum > 8)
		{
			printf("\n\n=====\nSOMETHING WENT WRONG WITH THE SUM. SUM = %d and HALF_SUM = %d\n=====\n\n", sum, half_sum);
		}

		if(sum == half_sum)
			matrix[i * num_elements + j] = (int)rand() % 2;
		else if(sum > half_sum)
			matrix[i * num_elements + j] = 1;
		else
			matrix[i * num_elements + j] = 0;

		if(matrix[i * num_elements + j] != prev_cell)
		{
			printf("Matrix thread #%d set position (%d, %d) to %d because it had a sum of %d and needed to beat %d\n", me, i, j, matrix[i * num_elements + j], sum, half_sum);
			printMatrix();
		}
		pthread_mutex_unlock(&mutex);
	}
	return (void*) 0;
}

/* Begin Problem Two code */
void printArray()
{
	printf("[");
	int i;
	for(i = 0; i < num_elements - 1; i++)
	{
		printf("%d, ", array[i]);
	}
	printf("%d]\n\n==========\n\n", array[i]);
}

void moveAndShift(int a[], int index, int newIndex)
{
	int temp = a[index];
	int i;
	for(i = index; i > newIndex; i--)
	{
		a[i] = a[i - 1];
	}
	a[newIndex] = temp;
}

int checkLeftAllLower(int a[], int n, int i)
{
	int isEverythingLower = 1;
	for(i = i - 1; i >= 0; i--)
	{
		if(a[i] > n)
		{
			isEverythingLower = 0;
			break;
		}
	}
	return isEverythingLower;
}

void* arrayChecker(void* arg)
{
	problem_two_running = 1;
	while(1)
	{
		int sorted = 1;
		pthread_mutex_lock(&mutex);
		int i;
		for(i = 1; i < num_elements; i++)
		{
			if(array[i - 1] > array[i])
			{
				sorted = 0;
				break;
			}
		}
		pthread_mutex_unlock(&mutex);
		if(sorted)
		{
			problem_two_running = 0;
			return (void*) 0;
		}

		sleep(0.1);
	}
	return (void*) 0;
}

void* arrayThread(void* arg)
{
	int me = (int) arg;
	while(problem_two_running)
	{
		int index = rand() % num_elements;
		int i = index;
		pthread_mutex_lock(&mutex);
		while(!checkLeftAllLower(array, array[index], i))
		{
			i--;
		}

		if(index != i)
		{
			printf("Thread #%d moved array[%d] = %d into index %d\n", me, index, array[index], i);
			moveAndShift(array, index, i);
			printArray();
		}
		pthread_mutex_unlock(&mutex);
	}
	return (void*) 0;
}

/* Begin helper functions for solving the problems and displaying UI */
void printHomeScreen()
{
	system("clear");
	printf("+=====================================================+\n");
	printf("|                                                     |\n");
	printf("|       Welcome to Matrix and Array Thread Demo       |\n");
	printf("|                                                     |\n");
	printf("+=================+=================+=================+\n");
	printf("|                 |                 |                 |\n");
	printf("| Problem One (1) | Problem Two (2) |   Exit (exit)   |\n");
	printf("|                 |                 |                 |\n");
	printf("+=================+=================+=================+\n");

	if(input_error)
	{
		printf("\n====== PLEASE MAKE SURE YOU ENTER VALID COMMANDS ======\n");
		input_error = 0;
	}
	else
	{
		printf("\n\n");
	}
        printf("\n\"1\", \"2\", \"exit\"> ");
}

void handleProblemOne()
{
	printf("Enter the number of threads to solve problem 1: ");
	char temp[16];
	fgets(temp, sizeof temp, stdin);
	fflush(stdin);
	int m = 0;
	m = atoi(temp);

	if(m < 1)
	{
		printf("Please enter a positive integer\n");
		return (void) 0;
	}

	pthread_t matrix_threads[m];
	pthread_t matrix_check;

	matrix = (int*) calloc(num_elements*num_elements, sizeof(int));

	int i;
	for(i = 0; i < num_elements * num_elements; i++)
	{
		matrix[i] = (rand() % 2);
	}

	printf("Starting matrix:\n");
	printMatrix();

	pthread_create(&matrix_check, 0, matrixChecker, NULL);
	for(i = 0; i < m; i++)
	{
		pthread_create(&matrix_threads[i], 0, matrixThread, (void*) i);
	}

	for(i = 0; i < m; i++)
	{
		pthread_join(matrix_threads[i], NULL);
	}
	pthread_join(matrix_check, NULL);
	printf("\nSolved problem one\n");
}

void handleProblemTwo()
{
	printf("Enter the number of threads to solve problem 2: ");
	char temp[16];
	fgets(temp, sizeof temp, stdin);
	fflush(stdin);
	int m = 0;
	m = atoi(temp);
	if(m < 1)
	{
		printf("Please enter a positive integer\n");
		return (void) 0;
	}

	pthread_t array_threads[m];
	pthread_t array_check;

	array = (int*) calloc(num_elements, sizeof(int));

	int i;
	for(i = 0; i < num_elements; i++)
	{
		array[i] = (rand() % (1000 - (-1000) + 1)) + (-1000);
	}

	printf("Starting array:\n");
	printArray();

	pthread_create(&array_check, 0, arrayChecker, NULL);
	for(i = 0; i < m; i++)
	{
		pthread_create(&array_threads[i], 0, arrayThread, (void*) i);
	}

	for(i = 0; i < m; i++)
	{
		pthread_join(array_threads[i], NULL);
	}
	pthread_join(array_check, NULL);
	printf("\nSolved problem two\n");
}

void waitExecution()
{
	printf("\nPress [Enter] to continue\n");
	fflush(stdout);
	getchar();

	/*
	int ch;
	do
		ch = fgetc(stdin);
	while(ch != EOF && ch != '\n');
	clearerr(stdin);
	*/
}

/* main function */
int main(int argc, char** argv)
{
	//arg parsing
	if(argc != 2)
	{
		printf("usage: %s <num_elements>\n", argv[0]);
		exit(-1);
	}

	num_elements = atoi(argv[1]);

	if(num_elements < 1 || num_elements > 1000)
	{
		printf("Please make sure you enter a reasonable number of elements (in the range [1, 100])\n");
		exit(-1);
	}

	//code initialization
	exit_code = 1;
	input_error = 0;
	srand(time(NULL));
	system("clear");

	//game loop
	while(exit_code)
	{
		printHomeScreen();

		fgets(input, sizeof(input), stdin);
		if(strncmp(input, "exit", 4) == 0)
		{
			exit_code = 0;
		}
		else if(strncmp(input, "1", 1) == 0 || strncmp(input, "one", 3) == 0)
		{
			handleProblemOne();
			waitExecution();
		}
		else if(strncmp(input, "2", 1) == 0 || strncmp(input, "two", 3) == 0)
		{
			handleProblemTwo();
			waitExecution();
		}
		else{
			input_error = 1;
		}
	}
	return 0;
}
