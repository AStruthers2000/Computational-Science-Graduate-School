#include "SineEnvelopeSineWave.h"

double SineEnvelopeSineWave::Evaluate(const vector<double>& input)
{
    double sum_term = 0.0;
    for(int i = 0; i < dimension - 1; i++)
    {
        const double x_i = input[i];
        const double x_ii = input[i + 1];

        const double x_i_2 = pow(x_i, 2);
        const double x_ii_2 = pow(x_ii, 2);
        
        const double numerator = pow(sin(x_i_2 + x_ii_2 - 0.5), 2);
        const double denominator = pow(1.0 + 0.001 * (x_i_2 + x_ii_2), 2);

        sum_term += 0.5 + numerator / denominator;
    }
    
    const double ans = -sum_term;
    
    return ans;
}
