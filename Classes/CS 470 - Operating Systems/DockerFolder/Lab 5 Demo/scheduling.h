#ifndef scheduling_h
#define scheduling_h

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <pthread.h>

/* Constant Definitions */

//swap to 0 when we want to disable debug info
#define DEBUG_PCB 1

#define PROCESS_SIZE 50
#define TIME_QUANTUM 2

#define PRIORITY_SORT 1
#define TIME_SORT 2

/* Global Variable Declarations */
typedef struct __attribute__((__packed__))
{
	char priority;
	char process_name[24];
	int process_id;
	char activity_status;
	int cpu_burst_time;
	int base_register;
	long limit_register;
	int num_files;
} pcb;

int** ready_queues;

int is_cpu_running;
int pause_execution;
int cores_waiting;

int num_cores;
int num_processes;

int* core_algs;
float* core_percents;

pcb* processes;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;


/* Function Declarations */
int parse_args(int argc, char** argv);
int load_processes(const char* filename);
void insert_sort(int core, int sort_type);

void print_ready_queues(void);
void print_ready_queue(int core);

void* core_priority(void* arg);
void* core_sjf(void* arg);
void* core_rr(void* arg);
void* core_fcfs(void* arg);
void* core_manager(void* arg);
void* priority_aging(void* arg);

#endif
