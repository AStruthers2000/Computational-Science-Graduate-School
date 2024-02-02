#include "RosenbrockSaddle.h"

double RosenbrockSaddle::Evaluate(const vector<double>& input)
{
    double ans = 0.0;

    for(int i = 0; i < dimension - 1; i++)
    {
        const double x_i = input[i];
        const double x_ii = input[i + 1];
        
        const double left_term = rosenbrock_constant * pow(pow(x_i, 2) - x_ii, 2);
        const double right_term = pow(1.0 - x_i, 2);

        ans += left_term + right_term;
    }
    
    return ans;
}
