#include "DeJong1.h"

double DeJong1::Evaluate(const vector<double>& input)
{
    double ans = 0.0;

    for(int i = 0; i < dimension; i++)
    {
        const double x_i = input[i];
        
        ans += pow(x_i, 2);
    }

    return ans;
}
