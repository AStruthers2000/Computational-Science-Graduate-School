#include <stdio.h>
#include <stdlib.h>
#include <string.h>


#include "struct_def.h"


int main()
{
	int shm_id;
	struct shared_struct *ptr;

	shm_id = shmget(SHM_KEY, sizeof(struct shared_struct), 0666);
	if(shm_id == -1)
	{
		printf("shmget failed\n");
		exit(-1);
	}

	ptr = (struct shared_struct*) shmat(shm_id, NULL, 0);
	if(ptr == (void*)-1)
	{
		printf("shmat failed\n");
		exit(-1);
	}

	//printf("ptr->player = %d\n", ptr->player);
	//printf("ptr->inpyt = %s\n", ptr->input);

	while(1)
	{
		while(ptr->player == 1);

		printf(">>> %s", ptr->input);

		if(strncmp(ptr->input, "exit", 4) == 0)
		{
			ptr->player = 1;
			break;
		}

		printf("> ");
		fgets(ptr->input, sizeof(ptr->input), stdin);
		fflush(stdin);

		ptr->player = 1;
	}

	printf("Child exiting\n");
	shmdt(ptr);
	return 0;
}
