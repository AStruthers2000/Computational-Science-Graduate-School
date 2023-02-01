#include "tictactoe.h"

int main()
{
	int shm_id;
	struct GameState* game;

	shm_id = shmget(SHM_KEY, sizeof(struct GameState), 0666);
	if(shm_id == -1)
	{
		//shmdt(game);
		//shmctl(shm_id, IPC_RMID, NULL);
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

	while(!game->init_ready);	//wait for host to initialize the game

	game->grid = (int*) shmat(shm_id, NULL, 0) + sizeof(struct GameState) + (game->grid_size * game->grid_size * sizeof(int));

	printf("Game initialized\n");
	printf("it is currently player %d's turn\n", game->player);
	printf("the game is running? %d\n", game->is_running);

	//initGame(game, game->grid_size, shm_id);
	printf("the first element in the grid is: %d\n", game->grid[0]);
	printGrid(game);

	while(game->is_running)
	{
		while(game->player == 1); //do nothing, it isn't my turn yet
		printGrid(game);

		if(!game->is_running)
		{
			break;
		}

		if(processTurn(game))
		{
			printGrid(game);
			game->player = 1;
		}
	}

	game->player = 1;

	if(game->winner == 1)
	{
		printf("Seems like player 1 won this round\n");
	}
	else if(game->winner == 2)
	{
		printf("Congrats player 2! You win!\n");
	}
	else
	{
		printf("Looks like the game ended in a draw\n");
	}

	printf("Client exiting\n");
	shmdt(game);
	shmctl(shm_id, IPC_RMID, NULL);
	return 0;
}
