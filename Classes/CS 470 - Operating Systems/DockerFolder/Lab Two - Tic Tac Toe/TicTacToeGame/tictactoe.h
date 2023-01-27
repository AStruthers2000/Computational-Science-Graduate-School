#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>
#include <sys/wait.h>
#include <sys/mman.h>

#define MAX_STR_SIZE 32
#define PLAYER_ONE "X "
#define PLAYER_TWO "O "

struct GameState
{
    char*** grid;
    int grid_size;
    int player;
    int counter;
} *game;

int is_running;


//functions
void initializeGame(void);
int convertInputToXY(const char* input, int *r, int *c);

void printGrid(void);
int getInput(int* x, int* y);
int updateGame(int* x, int* y);
int checkForWin(void);

int processTurn(void);
