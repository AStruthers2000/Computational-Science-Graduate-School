#include "Customer.h"

int main(int argc, char** argv)
{
	if(argc != 4)
	{
		strcpy(sServerIP, "127.0.0.1");
		nServerPort = 54321;
		nBurgersRequested = 10;
	}
	else
	{
		strcpy(sServerIP, argv[1]);
		nServerPort = atoi(argv[2]);
		nBurgersRequested = atoi(argv[3]);
	}

	if(nServerPort <= 0 || nBurgersRequested < 0)
	{
		printf("Please enter nonnegative values for the port and number of burgers\n");
		return -1;
	}

	printf("Attempting to connect to %s:%d, requesting a total of %d burgers\n", sServerIP, nServerPort, nBurgersRequested);


	int s_in = 0, n = 0;
	if((s_in = socket(AF_INET, SOCK_STREAM, 0)) < 0)
	{
		printf("\nError: Could not create socket\n");
		return -1;
	}

	struct sockaddr_in serv_addr;
	memset((char*) &serv_addr, '0', sizeof(serv_addr));
	serv_addr.sin_family = AF_INET;
	serv_addr.sin_port = htons(nServerPort);

	if(inet_pton(AF_INET, sServerIP, &serv_addr.sin_addr) <= 0)
	{
		printf("inet_pton error occured\n");
		return -1;
	}

	int nAttempts = 0;
	int bIsConnectionSuccessful = 0;
	while(nAttempts < TIMEOUT_ATTEMPTS)
	{
		if(connect(s_in, (struct sockaddr*) &serv_addr, sizeof(serv_addr)) < 0)
		{
			printf("\nError: Connect failed, retrying in %d seconds\n", TIMEOUT_TIME);
			nAttempts++;
			sleep(TIMEOUT_TIME);
		}
		else{
			bIsConnectionSuccessful = 1;
			break;
		}
	}
	if(!bIsConnectionSuccessful)
	{
		printf("\nFailed to connect\n");
		return -1;
	}

	printf("Connection to server made\n");
	srand(time(NULL));
	char buff[BUFFER_SIZE];
	char msg[BUFFER_SIZE];
	while(bBurgersRemaining)
	{
		snprintf(msg, sizeof(msg), "%d", (int)CLIENT_REQUESTS_BURGER);
		printf("Requesting a burger from burger joint\n");
		if(send(s_in, msg, strlen(msg), 0) < 0)
		{
			printf("\nError: failed to send server message\n");
			return -1;
		}

		n = recv(s_in, buff, sizeof(buff), 0);
		if(n < 0)
		{
			printf("\nError: failed to receive message from server\n");
			return -1;
		}
		else if(n == 0)
		{
			printf("Connection closed by server\n");
			return -1;
		}

		int code = atoi(buff);
		if(code == CLIENT_GETS_BURGER)
		{
			nSleepTime = (rand() % 3 + 1) * 2 - 1; //Generate either 1, 3, or 5
			printf("I just got a burger, but it will take %d minutes to eat it.\n", nSleepTime);
			sleep(nSleepTime);
			nBurgersServed++;
			printf("I have eaten %d burgers, and I still have %d burgers left to eat.\n", nBurgersServed, nBurgersRequested - nBurgersServed);
			if(nBurgersRequested - nBurgersServed <= 0)
			{
				printf("I am completely satisfied, so I will leave the burger joint\n");
				bBurgersRemaining = 0;
			}
		}
		else if(code == SERVER_OUT_OF_BURGERS)
		{
			printf("Burger joint ran out of burgers. Bummer\n");
			bBurgersRemaining = 0;
		}

		printf("\n\n=============== End of current interaction ===============\n\n");
	}

	if( n < 0)
	{
		printf("\nError: read error\n");
	}
	close(s_in);

	return 0;
}
