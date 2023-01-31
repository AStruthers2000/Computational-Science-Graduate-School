#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>

#define SHM_KEY 0x1234
#define MAX_STR_SIZE 64

struct shared_struct{
	int player;
	int spot;
	char input[MAX_STR_SIZE];
};

