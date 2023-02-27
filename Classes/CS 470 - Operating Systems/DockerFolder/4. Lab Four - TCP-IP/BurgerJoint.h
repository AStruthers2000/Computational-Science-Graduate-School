#include "shared_networking.h"

int nBurgersToBeProduced;
int nBurgersProduced;
int nChefs;
int nBurgersAvailable;
int nWaiters;

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
