#include "Schwefel.h"

double Schwefel::Evaluate(const vector<double>& input)
{
    double ans = schwefel_constant * dimension;

    double sum_term = 0.0;
    for(int i = 0; i < dimension; i++)
    {
        const double x_i = input[i];
        
        sum_term += -x_i * sin(sqrt(abs(x_i)));
    }
    ans -= sum_term;

    return ans;
}
