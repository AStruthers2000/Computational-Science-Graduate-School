#include "StretchVSineWave.h"

double StretchVSineWave::Evaluate(const vector<double>& input)
{
    double ans = 0.0;

    for(int i = 0; i < dimension - 1; i++)
    {
        const double x_i = input[i];
        const double x_ii = input[i + 1];

        const double x_i_2 = pow(x_i, 2);
        const double x_ii_2 = pow(x_ii, 2);

        const double left_root = pow(x_i_2 + x_ii_2, 0.25);
        const double right_root = pow(x_i_2 + x_ii_2, 0.1);

        ans += left_root * pow(50.0 * sin(right_root), 2) + 1.0;
    }
    
    return ans;
}
