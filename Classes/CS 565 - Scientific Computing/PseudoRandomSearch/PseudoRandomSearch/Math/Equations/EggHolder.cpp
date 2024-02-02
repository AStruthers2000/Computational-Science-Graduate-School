#include "EggHolder.h"

double EggHolder::Evaluate(const vector<double>& input)
{
    double ans = 0.0;

    for(int i = 0; i < dimension - 1; i++)
    {
        const double x_i = input[i];
        const double x_ii = input[i + 1];

        const double left_root = sqrt(abs(x_i - x_ii - eggholder_constant));
        const double right_root = sqrt(abs(x_ii + eggholder_constant + x_i/2.0));
        
        ans += -x_i * sin(left_root) - (x_ii + eggholder_constant) * sin(right_root);
    }
    
    return ans;
}
