#include "AckleyOne.h"

double AckleyOne::Evaluate(const vector<double>& input)
{
    double ans = 0.0;

    for(int i = 0; i < dimension - 1; i++)
    {
        const double x_i = input[i];
        const double x_ii = input[i + 1];

        const double x_i_2 = pow(x_i, 2);
        const double x_ii_2 = pow(x_ii, 2);

        const double root = sqrt(x_i_2 + x_ii_2);
        const double trig = 3.0 * (cos(2.0 * x_i) + sin(2.0 * x_ii));

        ans += ackleys_one_constant * root + trig;
    }
    
    return ans;
}
