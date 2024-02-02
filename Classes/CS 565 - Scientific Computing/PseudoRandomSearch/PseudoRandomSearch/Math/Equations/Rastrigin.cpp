#include "Rastrigin.h"

double Rastrigin::Evaluate(const vector<double>& input)
{
    double ans = rastrigin_constant * dimension;

    double sum_term = 0.0;
    for(int i = 0; i < dimension; i++)
    {
        const double x_i = input[i];

        sum_term += pow(x_i, 2) - rastrigin_constant * cos(PI_2 * x_i);
    }
    ans *= sum_term;
    
    return ans;
}
