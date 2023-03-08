#include "TestDLL.h"

EXPORT int add_numbers(int a, int b)
{
	printf("I have been provided two numbers, %d and %d\n", a, b);
	printf("\t%d + %d = %d\n", a, b, a+b);
	return a + b;
}

