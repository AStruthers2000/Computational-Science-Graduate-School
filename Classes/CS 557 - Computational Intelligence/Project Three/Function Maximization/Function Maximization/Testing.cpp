#include "Testing.h"

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>

float Test::f(float x, float y)
{
	return sin(M_PI * 10 * x + (10.0 / (1 + (y * y)))) + log(x * x + y * y);
}

float Test::calc_max(float& x, float& y)
{
	for (float i = MIN_X; i < MAX_X + step; i += step)
	{
		for (float j = MIN_Y; j < MAX_Y + step; j += step)
		{
			if (int(i) == i && int(j) == j)
			{
				std::cout << "(" << i << ", " << j << ")" << std::endl;
			}

			float v = f(i, j);
			if (v > func_max)
			{
				func_max = v;
				x = i;
				y = j;
			}
		}
	}
	return func_max;
}
