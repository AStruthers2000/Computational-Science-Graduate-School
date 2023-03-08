#include "Faster_kNN_Implementation.h"

int xInArray(const float* a, int size, float x)
{
	int i;
	for(i = 0; i < size; i++)
	{
		if(a[i] == x)
		{
			return 1;
		}
	}
	return 0;
}

float* kNN(const float( data, int nData, int k)
{
}

float IE_xy(const float* data_x, const float* data_y, int nData_x, int nData_y, int k)
{
	int unique_count = 0;
	float* yval = unique(data_y, nData_y, &unique_count);
}

float* unique(const float* data, int nData, int* unique_count)
{
	float* u = (float*) malloc(sizeof(float) * nData);
	//memset(&u, 0, sizeof(u));

	int i;
	for(i = 0; i < nData; i++)
	{
		if(!xInArray(u, (*unique_count), data[i]))
		{
			u[(*unique_count)++] = data[i];
			//(*unique_count)++;
		}
	}
	return u;
}

void sort(float* data, int nData)
{
}

// main only runs during testing
int main()
{
	// testing variables
	int fake_data_count = 10;
	float x[10] = {1, 3, 3, 5, 7, 18, 13, 20, 17, 7};
	float y[10] = {1, 2, 2, 1, 1, 1, 1, 2, 2, 2};

	//IE_xy(x, y, 2);

	int unique_count = 0;
	float* yval = unique(y, fake_data_count, &unique_count);

	print_array(y, fake_data_count, "Input array");
	print_array(yval, unique_count, "Unique elements");

	return 0;
}

void print_array(const float* a, int size, const char* msg)
{
	printf("%s: {", msg);
	int i;
	for(i = 0; i < size-1; i++)
	{
		printf("%4.4f ", a[i]);
	}
	printf("%4.4f}\n", a[size-1]);
}
