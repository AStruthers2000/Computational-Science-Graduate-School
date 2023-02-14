#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

//Definitions
#define MAX_STR_SIZE 32

//Variables
char input[MAX_STR_SIZE];
int exit_code;
int num_elements;
int input_error;
int* matrix;
int* array;
int problem_one_running;
int problem_two_running;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

//Functions
void printHomeScreen(void);
void waitExecution(void);
void handleProblemOne(void);
void handleProblemTwo(void);

void* matrixThread(void* arg);
void* matrixChecker(void* arg);
void* arrayThread(void* arg);
void* arrayChecker(void* arg);

int convertToSequence(int r, int c);
void moveAndShift(int a[], int index, int newIndex);
int checkLeftAllLower(int a[], int n, int i);

void printMatrix(void);
void printArray(void);
