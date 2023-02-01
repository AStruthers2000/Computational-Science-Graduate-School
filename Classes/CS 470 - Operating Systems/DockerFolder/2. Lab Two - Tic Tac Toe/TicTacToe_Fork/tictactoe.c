#include "tictactoe.h"

void initializeGame()
{
    is_running = 1;

    //game->grid = (char***)malloc(sizeof(char**) * game->grid_size);
    game->grid = mmap(NULL, sizeof(char***) * game->grid_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS,-1,0);

    int i;
    for(i = 0; i < game->grid_size; i++)
    {
        //game->grid[i] = (char**) malloc(sizeof(char*) * game->grid_size);

        game->grid[i] = mmap(NULL, sizeof(char**) * game->grid_size, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS,-1,0);
    }

    int r, c, count = 0;
    for(r = 0; r < game->grid_size; r++)
    {
        for(c = 0; c < game->grid_size; c++)
        {
            //char* g = (char*) malloc(sizeof(char) * MAX_STR_SIZE);
            char* g = mmap(NULL, sizeof(char) * MAX_STR_SIZE, PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS,-1,0);
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
            game->grid[r][c] = g;
            printf("Initializing grid[%d][%d] to %s\n", r, c, g);
        }
    }
}

int convertInputToXY(const char* input, int* r, int* c)
{
    int in = atoi(input);
    if(in > (game->grid_size * game->grid_size) || in <= 0)
    {
        printf("Invalid grid selection, please try again\n");
        return 0;
    }

    *r = (in - 1) / game->grid_size;
    *c = (in - 1) % game->grid_size;

    return 1;
}

int getInput(int* x, int* y)
{
    char input[MAX_STR_SIZE];
    printf("Player %d input: ", (game->player + 1));
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
    if(strcmp(game->grid[*r][*c], (char*) PLAYER_ONE) != 0 &&
       strcmp(game->grid[*r][*c], (char*) PLAYER_TWO) != 0)
    {
        game->grid[*r][*c] = (char*) (game->player == 0 ? PLAYER_ONE : PLAYER_TWO);
        //game->player = game->player == 0 ? 1 : 0;
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
    for(r = 0; r < game->grid_size; r++)
    {
        for(c = 0; c < (game->grid_size - 1); c++)
        {
            printf(" %s |", game->grid[r][c]);
        }
        printf(" %s \n", game->grid[r][game->grid_size - 1]);

        if(r < game->grid_size - 1)
        {
            int i;
            for(i = 0; i < (game->grid_size - 1); i++)
            {
                printf("----+");
            }
            printf("----\n");
        }
    }
    printf("\n");
}

/*
int checkForWin()
{
    int is_win = 0;
    int r, c;

    //horizontal
    for(r = 0; r < game->grid_size; r++)
    {
        int is_win_h = 1;
        //check each entry in row
        for(c = 1; c < game->grid_size; c++)
        {
            if(strcmp(game->grid[r][c], game->grid[r][c-1]) != 0)
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
    for(c = 0; c < game->grid_size; c++)
    {
        int is_win_v = 1;
        //check each column by going down the rows
        for(r = 1; r < game->grid_size; r++)
        {
            if(strcmp(game->grid[r][c], game->grid[r-1][c]) != 0)
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
    for(r = 1; r < game->grid_size; r++)
    {
        if(strcmp(game->grid[r][r], game->grid[r-1][r-1]) != 0)
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
    for(c = game->grid_size - 2; c >= 0; c--)
    {
        r = game->grid_size - c - 1;
        if(strcmp(game->grid[r][c], game->grid[r-1][c+1]) != 0)
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
*/
int processTurn()
{
    int x, y;
    if(getInput(&x, &y))
    {
        if(updateGame(&x, &y))
        {
            return 1;
        }
    }
    return 0;
}

int main(int argc, char** argv)
{
    if(argc != 2)
    {
        printf("Usage: %s <grid_size>\n", argv[0]);
        return -1;
    }

    game = mmap(NULL, sizeof(struct GameState), PROT_READ|PROT_WRITE, MAP_SHARED|MAP_ANONYMOUS, -1, 0);

    game->grid_size = atoi(argv[1]);
    if(game->grid_size < 3 || game->grid_size > 10)
    {
        printf("Invalid grid size, setting size to 3 (min: 3, max: 10)\n");
        game->grid_size = 3;
    }

    initializeGame();

    pid_t pid = fork();
    if(pid == 0)
    {
        while(is_running)
        {
            while (game->player == 0); //do nothing, it's not my turn

            if(processTurn())
            {
                game->player = 0;
            }
        }
    }
    else if(pid > 0)
    {
        while(is_running)
        {
            while (game->player == 1); //do nothing, it's not my turn

            printGrid();

            if(processTurn())
            {
                printGrid();
                game->player = 1;
            }
        }
    }

    printf("closing thing\n");
    munmap(game, sizeof(struct GameState));
    /*
    for(int i = 0; i < game->grid_size; i++)
    {
        printf("Freeing row\n");
        free(game->grid[i]);
    }
    printf("Freeing board\n");
    free(game->grid);
    //munmap(game, sizeof(struct GameState));
    */
}
