#pragma once

class Test 
{
private:
	const int MIN_X = 3;
	const int MAX_X = 10;
	const int MIN_Y = 4;
	const int MAX_Y = 8;

	const float step = 0.0001;

	float func_max = -1000000;

	float f(float x, float y);

public:
	float calc_max(float& x, float& y);

};