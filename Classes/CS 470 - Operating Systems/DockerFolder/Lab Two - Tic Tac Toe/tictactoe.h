#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <unistd.h>

#define MAX_STR_SIZE 32
#define PLAYER_ONE "X "
#define PLAYER_TWO "O "

int is_running;

struct GameSpace
{
	int grid_size;
	char ***grid;
	int cur_player;
	int winner;
} game;

struct GameSpace initializeGame(int size);
int convertInputToXY(const char* input, int *r, int *c);

void printGrid(void);
int getInput(int *x, int *y);
int updateGame(int *x, int *y);
int checkForWin(void);
