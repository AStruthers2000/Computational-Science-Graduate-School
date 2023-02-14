#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 100
#define NUM_ITERATIONS 1000000

long a;

pthread_mutex_t my_mut = PTHREAD_MUTEX_INITIALIZER;

void* Increment(void* arg)
{
	int thread_id = (int)arg;
	printf("I am starting an increment thread [+][%d]\n", thread_id);

	for(int i = 0; i < NUM_ITERATIONS; i++)
	{
		//pthread_mutex_lock(&my_mut);
		a++;
		//pthread_mutex_unlock(&my_mut);
	}

	printf("I am stopping an increment thread [+][%d]\n", thread_id);
	return (void*) 0;
}

void* Decrement(void* arg)
{
	int thread_id = (int)arg;
	printf("I am starting a  decrement thread [-][%d]\n", thread_id);

	for(int i = 0; i < NUM_ITERATIONS; i++)
	{
		//pthread_mutex_lock(&my_mut);
		a--;
		//pthread_mutex_unlock(&my_mut);
	}

	printf("I am stopping a  decrement thread [-][%d]\n", thread_id);
	return (void*) 0;
}

int main()
{

	pthread_t inc_threads[NUM_THREADS];
	pthread_t dec_threads[NUM_THREADS];

	printf("The value of a=%ld\n", a);

	for(int i = 0; i < NUM_THREADS; i++)
	{
		pthread_create(&inc_threads[i], 0, Increment, (void*)(int)i);
		pthread_create(&dec_threads[i], 0, Decrement, (void*)(int)i);
	}

	for(int i = 0; i < NUM_THREADS; i++)
	{
		pthread_join(inc_threads[i], NULL);
		pthread_join(dec_threads[i], NULL);
	}

	printf("The value of a=%ld\n", a);
}
