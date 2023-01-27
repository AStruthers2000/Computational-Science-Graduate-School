#include "tictactoe.h"

struct GameSpace initializeGame(int size)
{
	struct GameSpace g_space;

	char ***grid = (char***) malloc(sizeof(char**) * size);

	int i;
	for(i = 0; i < size; i++)
	{
		grid[i] = (char**) malloc(sizeof(char*) * size);
	}


	int r, c, count = 0;
	for(r = 0; r < size; r++)
	{
		for(c = 0; c < size; c++)
		{
			char* g = (char*) malloc(sizeof(char) * MAX_STR_SIZE);
			count++;
			if(count > 9)
			{
				sprintf(g, "%d", count);
			}
			else
			{
				g[0] = '0';
				g[1] = (char) (count + 48);
				g[2] = '\0';
			}
			grid[r][c] = g;
			printf("Initializing grid[%d][%d] to %s\n", r, c, g);
		}
	}

	g_space.grid_size = size;
	g_space.grid = grid;
	g_space.cur_player = 0;
	return g_space;
}

int convertInputToXY(const char* input, int* r, int* c)
{
	int in = atoi(input);
	if(in > (game.grid_size * game.grid_size) || in <= 0)
	{
		printf("Invalid grid selection, please try again\n");
		return 0;
	}

	*r = (in - 1) / game.grid_size;
	*c = (in - 1) % game.grid_size;

	return 1;
}

int getInput(int *x, int *y)
{
	char input[MAX_STR_SIZE];
	printf("Player %d input: ", (game.cur_player + 1));
	fgets(input, sizeof input, stdin);
	fflush(stdin);

	/*
	if(strncmp(input, "exit", 4) == 0)
	{
		is_running = 0;
		return 0;
	}
	*/

	int r, c;
	if(convertInputToXY(input, &r, &c))
	{
		*x = r;
		*y = c;
		return 1;
	}
	else
	{
		return 0;
	}
}

int updateGame(int* r, int* c)
{
	if(strcmp(game.grid[*r][*c], (char*) PLAYER_ONE) != 0 &&
	   strcmp(game.grid[*r][*c], (char*) PLAYER_TWO) != 0)
	{
		game.grid[*r][*c] = (char*) (game.cur_player == 0 ? PLAYER_ONE : PLAYER_TWO);
		game.cur_player = game.cur_player == 0 ? 1 : 0;
		return 1;
	}
	else
	{
		printf("Grid spot already has a value, please pick a different spot\n");
		return 0;
	}
}

void printGrid()
{
	printf("\n");
	int r, c;
	for(r = 0; r < game.grid_size; r++)
	{
		for(c = 0; c < (game.grid_size - 1); c++)
		{
			printf(" %s |", game.grid[r][c]);
		}
		printf(" %s \n", game.grid[r][game.grid_size - 1]);

		if(r < game.grid_size - 1)
		{
			int i;
			for(i = 0; i < (game.grid_size - 1); i++)
			{
				printf("----+");
			}
			printf("----\n");
		}
	}
	printf("\n");
}

int checkForWin()
{
	int is_win = 0;
	int r, c;

	//horizontal
	for(r = 0; r < game.grid_size; r++)
	{
		int is_win_h = 1;
		//check each entry in row
		for(c = 1; c < game.grid_size; c++)
		{
			if(strcmp(game.grid[r][c], game.grid[r][c-1]) != 0)
			{
				is_win_h = 0;
				break;
			}
		}
		if(is_win_h)
		{
			is_win = 1;
		}
	}

	//vertical
	for(c = 0; c < game.grid_size; c++)
	{
		int is_win_v = 1;
		//check each column by going down the rows
		for(r = 1; r < game.grid_size; r++)
		{
			if(strcmp(game.grid[r][c], game.grid[r-1][c]) != 0)
			{
				is_win_v = 0;
				break;
			}
		}
		if(is_win_v)
		{
			is_win = 1;
		}
	}

	//diagonal l to r
	int is_win_d = 1;
	for(r = 1; r < game.grid_size; r++)
	{
		if(strcmp(game.grid[r][r], game.grid[r-1][r-1]) != 0)
		{
			is_win_d = 0;
			break;
		}
	}
	if(is_win_d)
	{
		is_win = 1;
	}


	//diagonal r to l
	is_win_d = 1;
	for(c = game.grid_size - 2; c >= 0; c--)
	{
		r = game.grid_size - c - 1;
		if(strcmp(game.grid[r][c], game.grid[r-1][c+1]) != 0)
		{
			is_win_d = 0;
			break;
		}
	}
	if(is_win_d)
	{
		is_win = 1;
	}

	return is_win;
}


int main(int argc, char **argv)
{
	if(argc != 2)
	{
		printf("Usage: %s <grid_size>\n", argv[0]);
		exit(-1);
	}

	int grid_size = atoi(argv[1]);
	if(grid_size < 3 || grid_size > 10)
	{
		printf("Invalid grid size (min: 3, max: 10)\n\tDefaulting to 3\n");
		grid_size = 3;
	}

	//game = initializeGame(grid_size);

	game = mmap(NULL, sizeof(struct GameSpace), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);
	game = initializeGame(grid_size);

	int x, y;

	pid_t pid = fork();
	if(pid < 0)
	{
		printf("Fatal error when making fork, exiting\n");
		exit(-1);
	}
	else if(pid == 0)
	{
		while(1)
		{
			while(game.cur_player == 0); //do nothing because it is not my turn

			if(getInput(&x, &y))
			{
				if(updateGame(&x, &y))
				{
					if(checkForWin())
					{
						game.winner = 1;
						break;
					}
					printf("End of my turn, now to player 1\n");
					game.cur_player = 0;
				}
			}
		}

		if(game.winner)
		{
			printf("Player 2> See, the underdog can win!\n");
		}
		else
		{
			printf("Player 2> Oh... guess I lost\n");
		}
	}
	else{
		while(1)
		{
			while(game.cur_player == 1); //do nothing because it is not my turn
			printGrid(); //parent process is responsible for printing the contents of the grid

			if(getInput(&x, &y))
			{
				if(updateGame(&x, &y))
				{
					printGrid();
					if(checkForWin())
					{
						game.winner = 0;
						break;
					}
					printf("End of my turn, now to player 2\n");
					game.cur_player = 1;
				}
			}
		}

		if(game.winner)
		{
			printf("Player 1> It seems like I lost... :(\n");
		}
		else
		{
			printf("Player 1> I won! Yay!\n");
		}
	}


	int i;
	for(i = 0; i < grid_size; i++)
	{
		printf("Freeing dynamically allocated grid array %d\n", i+1);
		free(game->grid[i]);
	}
	free(game->grid);
}
