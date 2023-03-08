#ifndef TestDLL_h
#define TestDLL_h

#include <stdio.h>
#include <stdlib.h>

#define EXPORT __declspec(dllexport)

EXPORT int add_numbers(int a, int b);

#endif
