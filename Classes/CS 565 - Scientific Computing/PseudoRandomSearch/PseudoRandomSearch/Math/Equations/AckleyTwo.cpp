#include "AckleyTwo.h"

double AckleyTwo::Evaluate(const vector<double>& input)
{
    double ans = 0.0;

    for(int i = 0; i < dimension - 1; i++)
    {
        const double x_i = input[i];
        const double x_ii = input[i + 1];

        const double x_i_2 = pow(x_i, 2);
        const double x_ii_2 = pow(x_ii, 2);

        const double root = ackelys_two_e_02 * sqrt((x_i_2 + x_ii_2) / 2.0);
        const double trig = ackelys_two_e_05 * (cos(PI_2 * x_i) + sin(PI_2 * x_ii));

        
        ans += 20.0 + E - 20.0 / root - trig;
    }
    
    return ans;
}
