#include "BurgerJoint.h"

void* chef(void* arg)
{
	int me = (int)arg;
	while(nBurgersProduced < nBurgersToBeProduced)
	{
		pthread_mutex_lock(&mutex);
		nBurgersAvailable++;
		nBurgersProduced++;
		pthread_mutex_unlock(&mutex);
		int nCookTime = (rand()%2 + 1) * 2;
		printf("[CHEF]:       #%d just cooked a burger, there are now %d burgers available\n", me, nBurgersAvailable);
		sleep(nCookTime);
	}
	printf("[CHEF]:       #%d has finished cooking, all of today's burgers are done\n", me);
	return (void*) 0;
}

void* waiter(void* arg)
{
	int sock = *((int*) arg);
	free(arg);

	pthread_mutex_lock(&mutex);
	nWaiters++;
	int me = nWaiters;
	pthread_mutex_unlock(&mutex);

	printf("[WAITER]:     #%d is now waiting on new client\n", me);

	char buff[BUFFER_SIZE];
	char msg[BUFFER_SIZE];
	int n;
	while(nBurgersAvailable > 0 || nBurgersProduced < nBurgersToBeProduced)
	{
		n = recv(sock, buff, sizeof(buff), 0);
		if(n < 0)
		{
			printf("\nError: failed to receive message from client\n");
			break;
		}
		else if(n == 0)
		{
			printf("[WAITER]:     Customer has left waiter %d\n", me);
			break;
		}

		printf("[WAITER]:     #%d has received a request for a burger\n", me);

		//don't care what the client said, we know they are requesting burgers
		if(nBurgersAvailable > 0 || nBurgersProduced < nBurgersToBeProduced)
		{
			while(1)
			{
				pthread_mutex_lock(&mutex);
				if(nBurgersAvailable > 0)
				{
					snprintf(msg, sizeof(msg), "%d", (int)CLIENT_GETS_BURGER);
					//pthread_mutex_lock(&mutex);
					nBurgersAvailable--;
					printf("[WAITER]:     #%d has given a burger to the customer, there are still %d burgers available\n", me, nBurgersAvailable);
					pthread_mutex_unlock(&mutex);
					break;
				}
				pthread_mutex_unlock(&mutex);
			}
		}
		else
		{
			snprintf(msg, sizeof(msg), "%d", (int)SERVER_OUT_OF_BURGERS);
			printf("[WAITER]:     #%d has told its customer that there are no more burgers remaining.\n", me);
		}

		if(send(sock, msg, strlen(msg), 0) < 0)
		{
			printf("\nError: send to client failed\n");
			break;
		}
	}

	printf("[WAITER]:     #%d is done serving their customer\n", me);
	close(sock);
	pthread_exit(NULL);
}

int main(int argc, char** argv)
{
	if(argc != 3)
	{
		nBurgersToBeProduced = 25;
		nChefs = 2;
	}
	else
	{
		nBurgersToBeProduced = atoi(argv[1]);
		nChefs = atoi(argv[2]);
	}

	if(nBurgersToBeProduced <= 0 || nChefs <= 0)
	{
		printf("Please enter nonnegative values for number of burgers to be produced and the number of chefs\n");
		return -1;
	}

	printf("[RESTAURANT]: Opening restaurant... please wait\n");
	nWaiters = 0;
	nBurgersProduced = 0;
	nBurgersAvailable = 0;
	int i;
	for(i = 0; i < nChefs; i++)
	{
		pthread_t thread;
		pthread_create(&thread, NULL, chef, i+1);
	}
	sleep(1);

	int s_in = 0, conn = 0;
	if((s_in = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
		printf("\nError: Could not create socket\n");
		return -1;
	}

	struct sockaddr_in serv_addr;
	memset((char*) &serv_addr, '0', sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_addr.s_addr = htonl(INADDR_ANY);
	serv_addr.sin_port = htons(PORT);

	printf("[RESTAURANT]: Burger joint is now open and listening on 0.0.0.0:%d\n", PORT);

	bind(s_in, (struct sockaddr*) &serv_addr, sizeof(serv_addr));
	listen(s_in, TIMEOUT_TIME * TIMEOUT_ATTEMPTS);

	while(nBurgersAvailable > 0 || nBurgersProduced < nBurgersToBeProduced)
	{
		fd_set readfds;
		FD_ZERO(&readfds);
		FD_SET(s_in, &readfds);
		struct timeval timeout;
		timeout.tv_sec = 10;
		timeout.tv_usec = 0;

		int result = select(s_in+1, &readfds, NULL, NULL, &timeout);
		if(result < 0)
		{
			printf("\nError: failed to select socket\n");
			return -1;
		}
		else if(result == 0)
		{
			printf("[RESTAURANT]: No new customer in the restaurant for the past 10 seconds, trying again...\n");
		}
		else{
			int* client_socket = (int*) malloc(sizeof(int));
			*client_socket = accept(s_in, NULL, NULL);

			if(*client_socket < 0)
			{
				printf("\nError: failed to accept client\n");
				return -1;
			}

			pthread_t thread;
			if(pthread_create(&thread, NULL, waiter, client_socket) != 0)
			{
				printf("\nError: failed to create waiter thread\n");
				return -1;
			}
		}
	}

	printf("[RESTAURANT]: Burger joint is out of burgers, and will now close\n");

	close(s_in);

	return 0;
}
