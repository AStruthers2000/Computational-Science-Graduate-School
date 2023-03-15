#include "scheduling.h"

void insert_sort(int core, int sort_type)
{
	int real_jobs = 0;
	int i, j, key, job;
	for(i = 0; i < num_processes; i++)
	{
		if(ready_queues[core][i] != -1)
			real_jobs++;
		else
			break;
	}

	if(real_jobs < 2)
		return;
	int* sort_by_list = (int*) malloc(sizeof(int) * real_jobs);

	for(i = 0; i < real_jobs; i++)
	{
		pcb* job = &processes[ready_queues[core][i]];

		if(sort_type == PRIORITY_SORT)
			sort_by_list[i] = (int)(-job->priority);
		else if(sort_type == TIME_SORT)
			sort_by_list[i] = job->cpu_burst_time;
		else
			printf("Undefined sort key provided\n");
	}

	for(i = 1; i < real_jobs; i++)
	{
		key = sort_by_list[i];
		job = ready_queues[core][i];
		j = i - 1;

		while(j >= 0 && sort_by_list[j] > key)
		{
			sort_by_list[j+1] = sort_by_list[j];
			ready_queues[core][j+1] = ready_queues[core][j];
			j--;
		}
		sort_by_list[j + 1] = key;
		ready_queues[core][j + 1] = job;
	}
	free(sort_by_list);
	printf("\nFinished sorting core %d ready queue\n\n", core);
}

void pop_element(int core, int index)
{
	//pop the given element from the array
	int i;
	for(i = index + 1; i < num_processes; i++)
	{
		ready_queues[core][i - 1] = ready_queues[core][i];
	}
	ready_queues[core][num_processes - 1] = -1;
}

void* core_priority(void* arg)
{
	int core = (int) arg;
	printf("[Core %d, Priority]:        Now online\n", core);

	int executing_job_index = 0;
	int executing_job_id;
	while(is_cpu_running)
	{
		//pause execution if the core manager is load balancing
		if(pause_execution)
			cores_waiting++;
		while(pause_execution);

		if(ready_queues[core][0] == -1)
		{
			printf("[Core %d, Priority]:        Finished executing all jobs\n", core);
			printf("[Core %d, Priority]:        Requesting load balancing\n", core);
			pause_execution = 1;
		}
		else{

			executing_job_id = ready_queues[core][0];
			pcb* job = &processes[executing_job_id];
			printf("[Core %d, Priority]:        Executing job <%s>, priority %d, CPU burst time of %d\n", core, job->process_name, job->priority, job->cpu_burst_time);

			sleep(job->cpu_burst_time / 100.0);
			printf("[Core %d, Priority]:        Finished executing job <%s>\n", core, job->process_name);
			job->activity_status = 0;

			pop_element(core, 0);
		}
	}
	printf("Exiting Priority code %d\n", core);
	return (void*) 0;
}

void* core_sjf(void* arg)
{
	int core = (int) arg;
	printf("[Core %d, SJF]:             Now online\n", core);

	int executing_job_index = 0;
	int executing_job_id;
	while(is_cpu_running)
	{
		//pause execution if the core manager is load balancing
		if(pause_execution)
			cores_waiting++;
		while(pause_execution);

		if(ready_queues[core][0] == -1)
		{
			printf("[Core %d, SJF]:             Finished executing all jobs\n", core);
			printf("[Core %d, SJF]:             Requesting load balancing\n", core);
			pause_execution = 1;
		}
		else
		{
			if(ready_queues[core][0] == -1)
				break;

			executing_job_id = ready_queues[core][0];
			pcb* job = &processes[executing_job_id];
			printf("[Core %d, SJF]:             Executing job <%s>, CPU burst time of %d\n", core, job->process_name, job->cpu_burst_time);

			sleep(job->cpu_burst_time / 100.0);
			printf("[Core %d, SJF]:             Finished executing job <%s>\n", core, job->process_name);
			job->activity_status = 0;

			pop_element(core, 0);
		}
	}
	printf("Exiting SJF core %d\n", core);
	return (void*) 0;
}

void* core_rr(void* arg)
{
	int core = (int) arg;
	printf("[Core %d, Round Robin]:     Now online\n", core);

	int executing_job_index = 0;
	int executing_job_id;

	while(is_cpu_running)
	{
		//pause execution if the core manager is load balancing
		if(pause_execution)
			cores_waiting++;
		while(pause_execution);

		if(ready_queues[core][0] == -1)
		{
			printf("[Core %d, Round Robin]:     Finished executing all jobs\n", core);
			printf("[Core %d, Round Robin]:     Requesting load balancing\n", core);
			pause_execution = 1;
		}
		else
		{
			executing_job_id = ready_queues[core][executing_job_index];
			pcb* job = &processes[executing_job_id];
			printf("[Core %d, Round Robin]:     Executing job <%s>, remaining CPU burst of %d, time quantum of %d\n", core, job->process_name, job->cpu_burst_time, TIME_QUANTUM);

			sleep(TIME_QUANTUM / 100.0);
			job->cpu_burst_time -= TIME_QUANTUM;

			if(job->cpu_burst_time <= 0)
			{
				printf("[Core %d, Round Robin]:     Finished executing job <%s>\n", core, job->process_name);
				job->activity_status = 0;

				pop_element(core, executing_job_index);
				executing_job_index--;
			}
			else
				printf("[Core %d, Round Robin]:     Job <%s> still has %d remaining CPU burst time\n", core, job->process_name, job->cpu_burst_time);

			executing_job_index++;
			if(ready_queues[core][executing_job_index] == -1)
				executing_job_index = 0;
		}
	}
	printf("Exiting Round Robin core %d\n", core);
	return (void*) 0;
}

void* core_fcfs(void* arg)
{
	int core = (int) arg;
	printf("[Core %d, FCFS]:            Now online\n", core);

	int executing_job_index = 0;
	int executing_job_id;

	while(is_cpu_running)
	{
		//pause execution if the core manager is load balancing
		if(pause_execution)
			cores_waiting++;
		while(pause_execution);

		if(ready_queues[core][0] == -1)
		{
			printf("[Core %d, FCFS]:            Finished executing all jobs\n", core);
			printf("[Core %d, FCFS]:            Requesting load balancing\n", core);
			pause_execution = 1;
		}
		else
		{
			executing_job_id = ready_queues[core][0];
			pcb* job = &processes[executing_job_id];
			printf("[Core %d, FCFS]:            Executing job <%s>, CPU burst time of %d\n", core, job->process_name, job->cpu_burst_time);

			sleep(job->cpu_burst_time / 100.0);
			printf("[Core %d, FCFS]:            Finished executing job <%s>\n", core, job->process_name);
			job->activity_status = 0;

			pop_element(core, 0);
		}
	}
	printf("Exiting FCFC core %d\n", core);
	return (void*) 0;
}


void* core_manager(void* arg)
{
	is_cpu_running = 1;
	pause_execution = 0;

	while(is_cpu_running)
	{
		if(pause_execution)
		{
			printf("[CPU MANAGER]:             Need to do rebalancing\n");
			//wait for all cores to finish executing their current job before rebalancing
			while(cores_waiting != num_cores);
			cores_waiting = 0;

			pthread_mutex_lock(&mutex);
			printf("\n\n");
			printf("[CPU MANAGER]:             Ready queues before rebalancing\n\n");
			print_ready_queues();

			int i, j, jobs_remaining = 0;
			for(i = 0; i < num_cores; i++)
			{
				for(j = 0; j < num_processes; j++)
				{
					if(ready_queues[i][j] == -1)
						break;
					//ready_queues[i][j] = -1;
					jobs_remaining++;
				}
			}

			int* p = (int*) malloc(sizeof(int) * jobs_remaining);
			int p_index = 0;

			for(i = 0; i < num_cores; i++)
			{
				for(j = 0; j < num_processes; j++)
				{
					if(ready_queues[i][j] == -1)
						break;
					p[p_index] = ready_queues[i][j];
					ready_queues[i][j] = -1;
					p_index++;
				}
			}

			int processes_per_core = (int) (((float)jobs_remaining / (float)num_cores) + 0.5);
			p_index = 0;
			for(i = 0; i < num_cores; i++)
			{
				for(j = 0; j < processes_per_core; j++)
				{
					if(p_index >= jobs_remaining)
						break;

					ready_queues[i][j] = p[p_index];
					p_index++;
				}
			}

			for(i = 0; i < jobs_remaining - p_index; i++)
			{
				ready_queues[0][processes_per_core + i] = p[p_index + i];
			}

			//needs to now sort each ready queue
			for(i = 0; i < num_cores; i++)
			{
				if(core_algs[i] == 1)
				{
					insert_sort(i, PRIORITY_SORT);
				}
				else if(core_algs[i] == 2)
				{
					insert_sort(i, TIME_SORT);
				}
			}

			printf("\n[CPU MANAGER]:             Ready queues after rebalancing and sorting\n\n");
			print_ready_queues();
			printf("\n\n");
			pause_execution = 0;
			free(p);
			pthread_mutex_unlock(&mutex);
		}

		int all_jobs_complete = 1;

		pthread_mutex_lock(&mutex);
		int i;
		for(i = 0; i < num_processes; i++)
		{
			if(processes[i].activity_status != 0)
			{
				//printf("Still gotta complete process %d\n", processes[i].process_id);
				all_jobs_complete = 0;
				break;
			}
		}
		pthread_mutex_unlock(&mutex);

		sleep(1);

		if(all_jobs_complete)
		{
			printf("[CPU MANAGER]:              All processes completed\n");
			is_cpu_running = 0;
			pause_execution = 0;
		}

	}
	printf("Exiting CPU MANAGER\n");
	return (void*) 0;
}

void* priority_aging(void* arg)
{
	while(is_cpu_running)
	{
		sleep(10);
		if(!is_cpu_running)
			break;
		printf("\n");
		int i, j;
		for(i = 0; i < num_cores; i++)
		{
			if(core_algs[i] == 1)
			{
				for(j = 0; j < num_processes; j++)
				{
					if(ready_queues[i][j] == -1)
						break;
					pcb* job = &processes[ready_queues[i][j]];
					job->priority++;
					//printf("[AGING MANAGER]: increased priority of job <%s>\n", job->process_name);
				}
				printf("[AGING MANAGER]:           Increased priority of all jobs in core %d\n", i);
			}
		}
		printf("\n");
	}
	printf("Exiting AGING MANAGER\n");
	return (void*) 0;
}

void print_ready_queues()
{
	int i;
	for(i = 0; i < num_cores; i++)
	{
		print_ready_queue(i);
	}
}

void print_ready_queue(int core)
{
	int j;
	printf("Core %d has ready queue: {", core);
	for(j = 0; j < num_processes; j++)
	{
		if(ready_queues[core][j] == -1)
			break;
		printf("%d ", ready_queues[core][j]);
	}
	printf("}\n");
}

int parse_args(int argc, char** argv)
{
	num_cores = (argc - 2) / 2;
	core_algs = (int*) malloc(sizeof(int) * num_cores);
	core_percents = (float*) malloc(sizeof(float) * num_cores);
	float sum_percents = 0;
	int i;
	for(i = 2; i < argc; i+=2)
	{
		int index = (i / 2) - 1;
		core_algs[index] = atoi(argv[i]);
		if(core_algs[index] < 1 || core_algs[index] > 4)
		{
			printf("Please select either 1, 2, 3, or 4 for the scheduling algorithms\n");
			return 0;
		}
		core_percents[index] = atof(argv[i + 1]);
		sum_percents += core_percents[index];
	}
	if(sum_percents != 1)
	{
		printf("Core percent utilization should equal 1, instead got %f\n", sum_percents);
		return 0;
	}
	printf("There are supposed to be %d cores\n", num_cores);
	return 1;
}

int load_processes(const char* filename)
{
	FILE *f;
	if((f = fopen(filename, "rb")))
	{
		struct stat info;
		fstat(fileno(f), &info);
		off_t size = info.st_size;
		num_processes = size / PROCESS_SIZE;
		printf("We need an array of size %d to accomodate all processes\n", num_processes);

		//dynamically allocate the correct number of bytes in memory
		//should be the same number of bytes as the binary file
		processes = (pcb*) malloc(sizeof(pcb) * num_processes);

		char buff[PROCESS_SIZE];
		while(fread(buff, PROCESS_SIZE, 1, f) != 0)
		{
			//reading PROCESS_SIZE number of bytes at a time, cast those bytes to the pcb type
			pcb* process = (pcb*) &buff[0];

			//store the current process at the appropriate index of the dynamically allocated array of processes
			processes[process->process_id] = *process;
		}
		//gotta remember to close our files
		fclose(f);
		return 1;
	}
	return 0;
}

//Priority = 1, SJF = 2, RR = 3, FCFS = 4
int main(int argc, char** argv)
{
	if(argc < 4 || argc % 2 != 0)
	{
		printf("usage: <binary_file_to_read> <scheduling_alg_1> <core_1_percent> ... <scheduling_alg_n> <core_n_percent>\n");
		return -1;
	}

	if(!parse_args(argc, argv))
	{
		printf("Something went wrong when parsing args, see above error messages\n");
		return -1;
	}

	if(!load_processes(argv[1]))
	{
		printf("Failed to read provided file and load them in to the processes array\n");
		return -1;
	}

	//inside of the ready queue, we are just going to store an int process_id for easy sorting
	ready_queues = (int**) malloc(sizeof(int*) * num_cores);
	int i, j;
	for(i = 0; i < num_cores; i++)
	{
		//give each core the option to hold all of the cores
		ready_queues[i] = (int*) malloc(sizeof(int) * num_processes);
		for(j = 0; j < num_processes; j++)
		{
			ready_queues[i][j] = -1;
		}
	}

	int assigned_processes = 0;
	for(i = 0; i < num_cores - 1; i++)
	{
		int needs_assigning = (int)(num_processes * core_percents[i]);
		printf("Core %d needs %d processes\n", i, needs_assigning);
		for(j = 0; j < needs_assigning; j++)
		{
			ready_queues[i][j] = processes[assigned_processes].process_id;
			assigned_processes++;
		}
	}
	printf("Core %d gets the remaining %d processes\n", i, (num_processes - assigned_processes));
	for(j = 0; j < (num_processes - assigned_processes); j++)
	{
		ready_queues[num_cores - 1][j] = processes[j + assigned_processes].process_id;
	}

	print_ready_queues();

	printf("\n\n==================== Starting Simulation ====================\n\n\n");

	pthread_t cores[num_cores];
	pthread_t manager;
	pthread_t aging;
	int using_aging = 0;

	pthread_create(&manager, 0, core_manager, NULL);
	for(i = 0; i < num_cores; i++)
	{
		if(core_algs[i] == 1)
		{
			insert_sort(i, PRIORITY_SORT);
			pthread_create(&cores[i], 0, core_priority, (void*) i);
			if(!using_aging)
			{
				pthread_create(&aging, 0, priority_aging, NULL);
				using_aging = 1;
			}
		}
		else if(core_algs[i] == 2)
		{
			insert_sort(i, TIME_SORT);
			pthread_create(&cores[i], 0, core_sjf, (void*) i);
		}
		else if(core_algs[i] == 3)
		{
			pthread_create(&cores[i], 0, core_rr, (void*) i);
		}
		else if(core_algs[i] == 4)
		{
			pthread_create(&cores[i], 0, core_fcfs, (void*) i);
		}
		else
		{
			printf("This should never be printed, core algs error checking should have been handled already\n");
			return -1;
		}
	}

	for(i = 0; i < num_cores; i++)
	{
		pthread_join(cores[i], NULL);
	}
	if(using_aging)
		pthread_join(aging, NULL);
	pthread_join(manager, NULL);
	printf("\n\n==================== Finished executing all processes ====================\n\n");
	free(ready_queues);
	free(core_algs);
	free(core_percents);
	free(processes);
/*
#if DEBUG_PCB
	for(i = 0; i < num_processes; i++)
	{
		pcb process = processes[i];

		printf("%s has a PCB of:\n", process.process_name);
		printf("\tpriority:        %d\n", process.priority);
		printf("\tprocess id:      %d\n", process.process_id);
		printf("\tactivity status: %d\n", process.activity_status);
		printf("\tcpu burst time:  %d\n", process.cpu_burst_time);
		printf("\tbase register:   %d\n", process.base_register);
		printf("\tlimit register:  %ld\n", process.limit_register);
		printf("\tnumber of files: %d\n", process.num_files);
	}
#endif
*/

	return 0;
}
