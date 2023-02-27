//#ifndef shared_network
//#define shared_network

#include <sys/socket.h>
#include <sys/types.h>
#include <sys/select.h>
#include <netinet/in.h>
#include <netdb.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unistd.h>
#include <errno.h>
#include <arpa/inet.h>
#include <time.h>
#include <pthread.h>

#define PORT 54321
#define BUFFER_SIZE 1024
#define TIMEOUT_TIME 5 		//5 seconds before client generates a timeout signal
#define TIMEOUT_ATTEMPTS 12 	//allow the client to try reconnecting 12 times (one minute) before kicking them out

#define CLIENT_GETS_BURGER 1
#define CLIENT_REQUESTS_BURGER 2
#define SERVER_OUT_OF_BURGERS 0
#define SERVER_EXITING -1

#define STR_LEN 64

struct BurgerRequest{
	int amount;
};

//#endif
