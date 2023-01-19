#include "main.h"

int add(int a, int b)
{
	return a+b;
}

int main()
{
	int x=5, y=6, z;
	z = add(x, y);
	printf("The resut of adding %d and %d together is %d\n", x, y, z);
}
