#include "tictactoe.h"

void initGame(struct GameState* state, int size, int id)
{
	/*
	printf("Initializing game\n");
	state->grid_size = size;

	//state->grid = (char***)malloc(sizeof(char**) * state->grid_size);
	//state->grid = mmap(NULL, sizeof(char***) * state->grid_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS,-1,0);
	state->grid = (char***) shmat(id, NULL, 0) + sizeof(struct GameState);// * state->grid_size;

	int i;
	for(i = 0; i < state->grid_size; i++)
	{
		//state->grid[i] = (char**) malloc(sizeof(char*) * state->grid_size);
		//state->grid[i] = mmap(NULL, sizeof(char**) * state->grid_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS,-1,0);
		state->grid[i] = (char**) shmat(id, NULL, 0) + sizeof(struct GameState) + (state->grid_size * sizeof(char*));
	}

	int r, c, count = 0;
	for(r = 0; r < state->grid_size; r++)
	{
		for(c = 0; c < state->grid_size; c++)
		{
			//state->grid[r][c] = (char*) shmat(id, NULL, 0) + sizeof(struct GameState) + (state->grid_size * state->grid_size * sizeof(char*));
			count++;
			//char* g = (char*) malloc(sizeof(char) * MAX_STR_SIZE);
			//char* g = mmap(NULL, sizeof(char) * MAX_STR_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS,-1,0);
			char* g = (char*) shmat(id, NULL, 0) + sizeof(struct GameState) + (state->grid_size * state->grid_size * sizeof(char*) * (i+1) * (c+1));

			//char g[MAX_STR_SIZE];
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
			//state->grid[r][c] = (char*) shmat(id, NULL, 0) + sizeof(struct GameState) + (state->grid_size * state->grid_size * sizeof(char*) * (state->grid_size * r + c));
			//if(state->grid[r][c] == (void*)-1)
			//{
			//	printf("shmat error when building grid\n");
			//}
			//strcpy(state->grid[r][c], g);
			state->grid[r][c] = g;
			printf("Initializing grid[%d][%d] to %s -> grid[%d][%d] = %s\n", r, c, g, r, c, state->grid[r][c]);

		}
	}
	*/

	state->grid_size = size;
	state->grid = (int*) shmat(id, NULL, 0) + sizeof(struct GameState) + (state->grid_size * state->grid_size * sizeof(int));
	int i, j;
	for(i = 0; i < state->grid_size; i++)
	{
		for(j = 0; j < state->grid_size; j++)
		{
			state->grid[i * state->grid_size + j] = i * state->grid_size + j + 1;
		}
	}

	//state->grid[0][0] = "01\0";

	state->is_running = 1;
	state->player = 1;
	state->winner = 0;
	state->init_ready = 1;
}

int convertInputToXY(const struct GameState* state, const char* input, int* r, int* c)
{
	int in = atoi(input);
	if(in > (state->grid_size * state->grid_size) || in <= 0)
	{
		printf("Invalid grid selection, please try again\n");
		return 0;
	}

	*r = (in - 1) / state->grid_size;
	*c = (in - 1) % state->grid_size;

	return 1;
}

int getInput(struct GameState* state, int* x, int* y)
{
	char input[MAX_STR_SIZE];
	printf("Player %d input: ", (state->player));
	fgets(input, sizeof input, stdin);
	fflush(stdin);

	if(strncmp(input, "exit", 4) == 0)
	{
		state->is_running = 0;
		return 0;
	}

	int r, c;
	if(convertInputToXY(state, input, &r, &c))
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

int updateGame(struct GameState* state, int* r, int* c)
{
	int i = *r * state->grid_size + *c;
	if(state->grid[i] != -1 && state->grid[i] != -2)
	{
		state->grid[i] = -state->player;
		return 1;
	}
	else
	{
		printf("Grid spot already has a value, please pick a different spot\n");
		return 0;
	}
}

void printGrid(const struct GameState* state)
{
	printf("\n");

	int r, c;
	for(r = 0; r < state->grid_size; r++)
	{
		for(c = 0; c < state->grid_size; c++)
		{
			char g[MAX_STR_SIZE];
			g[0] = '\0';

			int i = r * state->grid_size + c;
			if(state->grid[i] == -1 || state->grid[i] == -2)
			{
				snprintf(g, sizeof(g), "%s", state->grid[i] == -1 ? PLAYER_ONE : PLAYER_TWO);
			}
			else if(state->grid[i] < 10)
			{
				snprintf(g, sizeof(g), "0%d", state->grid[i]);
			}
			else
			{
				snprintf(g, sizeof(g), "%d", state->grid[i]);
			}


			if(c == state->grid_size - 1)
			{
				printf(" %s \n", g);
			}
			else
			{
				printf(" %s |", g);
			}
		}

		if(r < state->grid_size - 1)
		{
			int i;
			for(i = 0; i < (state->grid_size - 1); i++)
			{
				printf("----+");
			}
			printf("----\n");
		}
	}
	printf("\n");
}

int checkForWin(const struct GameState* state)
{
	int is_win = 0;
	int r, c, i;

	//horizontal
	for(r = 0; r < state->grid_size; r++)
	{
		int is_win_h = 1;

		//check each entry in row
		for(c = 1; c < state->grid_size; c++)
		{
			i = r * state->grid_size + c;
			if(state->grid[i] != state->grid[i-1])
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
	for(r = 0; r < state->grid_size; r++)
	{
		int is_win_v = 1;

		//check each column by going down the rows
		for(c = 1; c < state->grid_size; c++)
		{
			i = r + c * state->grid_size;
			if(state->grid[i] != state->grid[r])
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
	for(r = 1; r < state->grid_size; r++)
	{
		i = r * state->grid_size + r;
		if(state->grid[i] != state->grid[0])
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
	for(r = 1; r < state->grid_size; r++)
	{
		i = r * state->grid_size + state->grid_size - 1 - r;
		if(state->grid[i] != state->grid[state->grid_size - 1])
		{
			is_win_d = 0;
			break;
		}
	}
	if(is_win_d)
	{
		is_win = 1;
	}

	if(!is_win)
	{
		if(state->total_turns == state->grid_size * state->grid_size)
		{
			return -1;
		}
	}

	return is_win;
}

int processTurn(struct GameState* state)
{
	int x, y;
	if(getInput(state, &x, &y))
	{
		if(updateGame(state, &x, &y))
		{
			state->total_turns++;
			return 1;
		}
	}
	return 0;
}
