#ifndef MATTHREAD_H
#define MATTHREAD_H

typedef struct
{
	long double** base;
	long double** prev;
	long double** result;
	int dimension;
	int index;
} thread_data;

void* multiply_row(void* arg);
void* multiply_col(void* arg);
void* multiply_element(void* arg);

#endif
