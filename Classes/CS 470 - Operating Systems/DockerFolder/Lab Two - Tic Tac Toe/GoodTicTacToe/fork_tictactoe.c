#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/sem.h>

#define SHM_KEY 0x1234
#define SEM_KEY 0x5678

int grid_size;
int semid;
int shmid;
char *board;

void drawBoard()
{
	int i, j;
	for(i = 0; i < grid_size; i++)
	{
		for(j = 0; j < grid_size; j++)
		{
			printf("%c ", board[i * grid_size + j]);
		}
		printf("\n");
	}
}

void input(char player)
{
	int a;
	printf("%c: enter the number of the field: ", player);
	scanf("%d", &a);

	int x, y;
	x = (a - 1) / grid_size;
	y = (a - 1) % grid_size;

	struct sembuf sb;
	sb.sem_num = 0;
	sb.sem_op = -1;
	sb.sem_flg = SEM_UNDO;
	semop(semid, &sb, 1);

	board[x * grid_size + y] = player;

	sb.sem_op = 1;
	semop(semid, &sb, 1);
}

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		printf("Usage: %s <grid_size>\n", argv[0]);
		return 1;
	}

	grid_size = atoi(argv[1]);
	if(grid_size < 3 || grid_size > 10)
	{
		printf("Invalid grid size, setting size to 3 (min: 3, max: 10)\n");
	}

	shmid = shmget(SHM_KEY, grid_size * grid_size, IPC_CREAT | 0666);
	board = shmat(shmid, NULL, 0);

	semid = semget(SEM_KEY, 1, IPC_CREAT | 0666);
	semctl(semid, 0, SETVAL, 1);

	pid_t pid = fork();
	if(pid == 0)
	{
		while(1)
		{
			input('O');
			if
	}
}
