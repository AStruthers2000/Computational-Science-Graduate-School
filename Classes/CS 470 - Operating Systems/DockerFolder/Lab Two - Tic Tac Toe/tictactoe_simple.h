#include <stdio.h>
#include <stdlib.h>
#include <string.h>

//definitions
#define MAX_STR_SIZE 32
#define PLAYER_ONE "X "
#define PLAYER_TWO "O "

//variables
int is_running;
int grid_size;
char*** grid;
int player;


//functions
void initializeGame(void);
int convertInputToXY(const char* input, int *r, int *c);

void printGrid(void);
int getInput(int* x, int* y);
int updateGame(int* x, int* y);
int checkForWin(void);
