#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>

int main()
{
	printf("Weird demo\n");

	int globalVar = 0;
	pid_t pid = fork();
	if(pid == 0)
	{
		printf("Before entering the child process, variable is: %d\n", globalVar);
		//printf("I am in the child process\n");
		printf("Me, the child, has pid: %d and my parent pid is %d\n", getpid(), getppid());
		globalVar = 10;
	}
	else
	{
		if(pid > 0)
		{
			wait(NULL);
			printf("Before entering the parent process, variable is %d\n", globalVar);
			printf("Parent pid is: %d and my parent pid is: %d\n", getpid(), getppid());
			globalVar = 1;
		}
		else
		{
			printf("Error in process creating\n");
			exit(-1);
		}
	}
	printf("Variable is %d\n", globalVar);
	printf("Weird demo done\n");
	return 0;
}
