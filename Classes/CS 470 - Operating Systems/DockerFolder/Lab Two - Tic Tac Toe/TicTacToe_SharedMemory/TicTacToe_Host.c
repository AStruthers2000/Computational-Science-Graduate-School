#include "tictactoe.h"

int main(int argc, char** argv)
{
	//validating grid size from command line argument
	if(argc != 2)
	{
		printf("usage: %s <grid_size>\n", argv[0]);
		exit(-1);
	}

	int size = atoi(argv[1]);
	if(size < 3 || size > 10)
	{
		printf("Invalid grid size, defaulting to 3\n");
		size = 3;
	}

	//setting up shared memory of struct GameState* game defined in tictactoe.h
	int shm_id;
	struct GameState* game;

	shm_id = shmget(SHM_KEY, sizeof(struct GameState), IPC_CREAT | 0666);
	//shm_id = shmget(SHM_KEY, 4096, IPC_CREAT | 0666);

	if(shm_id == -1)
	{
		printf("shmget failed\n");
		perror("shmget");
		exit(-1);
	}

	game = (struct GameState*) shmat(shm_id, NULL, 0);
	if(game == (void*)-1)
	{
		printf("shmat failed\n");
		exit(-1);
	}

	initGame(game, size, shm_id);
	printf("Initialized game successfully\n");

	//this line segfaults
	//printf("%s\n", game->grid[0][0]);

	//game->grid[0] = 99;

	while(game->is_running)
	{
		while(game->player == 2); //do nothing, it isn't my turn yet
		printGrid(game);

		int win;
		if((win = checkForWin(game)) > 0)
		{
			//printf("Congrats player 1! You win!\n");
			game->winner = 2;
			game->is_running = 0;
		}
		else if(win == -1)
		{
			game->is_running = 0;
		}

		if(!game->is_running)
		{
			break;
		}

		if(processTurn(game))
		{
			printGrid(game);
			if((win = checkForWin(game)) > 0)
			{
				game->winner = 1;
				game->is_running = 0;
				break;
			}
			else if(win == -1)
			{
				game->is_running = 0;
				break;
			}
			game->player = 2;
		}
	}

	game->player = 2;

	if(game->winner == 1)
	{
		printf("Congrats player 1! You win!\n");
	}
	else if(game->winner == 2)
	{
		printf("Seems like player 2 won this round\n");
	}
	else{
		printf("Looks like the game ended in a draw\n");
	}

	printf("Host exiting\n");
	shmdt(game);
	shmctl(shm_id, IPC_RMID, NULL);
	return 0;
}

