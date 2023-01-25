#include "tictactoe_simple.h"

void initializeGame()
{
	is_running = 1;
	player = 1;

	grid = (char***)malloc(sizeof(char**) * grid_size);

	int i;
	for(i = 0; i < grid_size; i++)
	{
		grid[i] = (char**) malloc(sizeof(char*) * grid_size);
	}

	int r, c, count = 0;
	for(r = 0; r < grid_size; r++)
	{
		for(c = 0; c < grid_size; c++)
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

	printGrid();
}

int convertInputToXY(const char* input, int* r, int* c)
{
	int in = atoi(input);
	if(in > (grid_size * grid_size) || in <= 0)
	{
		printf("Invalid grid selection, please try again\n");
		return 0;
	}

	*r = (in - 1) / grid_size;
	*c = (in - 1) % grid_size;

	return 1;
}

int getInput(int* x, int* y)
{
	char input[MAX_STR_SIZE];
	printf("Player %d input: ", player);
	fgets(input, sizeof input, stdin);
	fflush(stdin);
	if(strncmp(input, "exit", 4) == 0)
	{
		is_running = 0;
		return 0;
	}

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
	if(strcmp(grid[*r][*c], (char*) PLAYER_ONE) != 0 &&
	   strcmp(grid[*r][*c], (char*) PLAYER_TWO) != 0)
	{
		grid[*r][*c] = (char*) (player == 1 ? PLAYER_ONE : PLAYER_TWO);
		player = player == 1 ? 2 : 1;
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
	for(r = 0; r < grid_size; r++)
	{
		for(c = 0; c < (grid_size - 1); c++)
		{
			printf(" %s |", grid[r][c]);
		}
		printf(" %s \n", grid[r][grid_size - 1]);

		if(r < grid_size - 1)
		{
			int i;
			for(i = 0; i < (grid_size - 1); i++)
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
	//int is_win_h = 1, is_win_v = 1, is_win_d = 1;
	int is_win = 0;
	int r, c;

	//horizontal
	for(r = 0; r < grid_size; r++)
	{
		int is_win_h = 1;
		//check each entry in row
		for(c = 1; c < grid_size; c++)
		{
			if(strcmp(grid[r][c], grid[r][c-1]) != 0)
			{
				//printf("Failed horizontal check\n");
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
	for(c = 0; c < grid_size; c++)
	{
		int is_win_v = 1;
		//check each column by going down the rows
		for(r = 1; r < grid_size; r++)
		{
			if(strcmp(grid[r][c], grid[r-1][c]) != 0)
			{
				//printf("Failed vertical check\n");
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
	for(r = 1; r < grid_size; r++){
		if(strcmp(grid[r][r], grid[r-1][r-1]) != 0)
		{
			//printf("Failed l to r diagonal check\n");
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
	for(c = grid_size - 2; c >= 0; c--)
	{
		r = grid_size - c - 1;
		//printf("grid[%d][%d] = %s\n", r, c, grid[r][c]);
		if(strcmp(grid[r][c], grid[r-1][c+1]) != 0)
		{
			//printf("Failed r to l diagonal check\n");
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

int main(int argc, char** argv)
{
	if(argc != 2)
	{
		printf("Invalid arguments, please enter a single grid dimension\n");
		exit(-1);
	}

	grid_size = atoi(argv[1]);
	if(grid_size <= 0 || grid_size > 10)
	{
		printf("Bad size argument detected, defaulting to standard grid size of 3\n");
		grid_size = 3;
	}

	//printf("grid size: %d\n", grid_size);

	initializeGame();

	while(is_running)
	{
		int x, y;
		if(getInput(&x, &y))
		{
			if(updateGame(&x, &y))
			{
				printGrid();
				if(checkForWin())
				{
					printf("Congrats player %d, you won!\n", player == 1 ? 2 : 1);
					is_running = 0;
				}
			}
		}
	}

	printf("Thanks for playing!\n");
	int i;
	for(i = 0; i < grid_size; i++)
	{
		printf("Freeing dynamically allocated grid array %d\n", i+1);
		free(grid[i]);
	}
	return 0;
}
