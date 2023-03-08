#pragma once

#include <stdio.h>
#include <stdlib.h>

#ifdef MATHLIBRARY_EXPORTS
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __declspec(dllimport)
#endif



extern "C" EXPORT void add_numbers(const int* a, const int* b, int* sum);
