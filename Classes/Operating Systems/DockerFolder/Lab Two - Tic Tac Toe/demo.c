#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <time.h>

int main()
{
	printf("Weird demo\n");

	pid_t pid = fork();
	if(pid == 0)
	{
		//printf("I am in the child process\n");
		printf("Me, the child, has pid: %d and my parent pid is %d\n", getpid(), getppid());
		sleep(500);
	}
	else
	{
		if(pid > 0)
		{
			//printf("I am the parent process\n");
			printf("Parent pid is: %d and my parent pid is: %d\n", getpid(), getppid());
			sleep(500);
		}
		else
		{
			printf("Error in process creating\n");
			exit(-1);
		}
	}
	printf("Weird demo done\n");
	return 0;
}
