#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/types.h>

#define SHM_KEY 0x12345
#define MAX_STR_SIZE 16
#define PLAYER_ONE "X "
#define PLAYER_TWO "O "
#define GAMESTATE_SIZE (size_t)1024

struct GameState
{
	int grid_size;
	int player;
	int winner;
	int init_ready;
	int is_running;
	int total_turns;
	int* grid;
};

void initGame(struct GameState* state, int size, int id);

int processTurn(struct GameState* state);
int convertInputToXY(const struct GameState* state, const char* input, int* r, int* c);

void printGrid(const struct GameState* state);
int getInput(struct GameState* state, int* x, int* y);
int updateGame(struct GameState* state, int* x, int* y);
int checkForWin(const struct GameState* state);


