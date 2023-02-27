#include "shared_networking.h"

char sServerIP[STR_LEN];
int nServerPort;
int nBurgersRequested;
int nBurgersServed = 0;
int bBurgersRemaining = 1;
int nSleepTime;
