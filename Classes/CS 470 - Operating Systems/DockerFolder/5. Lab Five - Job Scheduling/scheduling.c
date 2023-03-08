#include "scheduling.h"

struct __attribute__((__packed__)) pcb
{
	char priority;
	char process_name[24];
	int process_id;
	char activity_status;
	int cpu_burst_time;
	int base_register;
	long limit_register;
	int num_files;
};


//Priority = 1, SJF = 2, RR = 3, FCFS = 4
int main(int argc, char** argv)
{
	//I want to read the binary file provided in argv[1]
	//load that into struct __attribute__(__packed__) pcb

	if(argc < 2 && argc % 2 != 0)
	{
		printf("usage: <binary_file_to_read> \n");
		return -1;
	}

	char buff[50];
	FILE *f;
	if((f = fopen(argv[1], "rb")))
	{
		struct stat info;
		fstat(fileno(f), &info);
		off_t size = info.st_size;
		int processCount = size / PROCESS_SIZE;
		printf("We need an array of size %d to accomodate all processes\n", processCount);
		while(fread(buff, 50, 1, f) != 0)
		{
			struct pcb *process = (struct pcb*) &buff[0];
			printf("%s has a PCB of:\n", process->process_name);
			printf("\tpriority:        %d\n", process->priority);
			printf("\tprocess id:      %d\n", process->process_id);
			printf("\tactivity status: %d\n", process->activity_status);
			printf("\tcpu burst time:  %d\n", process->cpu_burst_time);
			printf("\tbase register:   %d\n", process->base_register);
			printf("\tlimit register:  %ld\n", process->limit_register);
			printf("\tnumber of files: %d\n", process->num_files);
		}
		fclose(f);
	}
	return 0;
}
