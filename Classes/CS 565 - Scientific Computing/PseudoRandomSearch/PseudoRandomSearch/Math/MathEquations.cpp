#include "MathEquations.h"
#include <cmath>

double MathEquations::Schwefel(double input[dimension])
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

double MathEquations::DeJong1(double input[dimension])
{
    double ans = 0.0;

    for(int i = 0; i < dimension; i++)
    {
        const double x_i = input[i];
        
        ans += pow(x_i, 2);
    }

    return ans;
}

double MathEquations::RosenbrockSaddle(double input[dimension])
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

double MathEquations::Rastrigin(double input[dimension])
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

double MathEquations::Griewangk(double input[dimension])
{
    double sum_term = 0.0;
    double product_term = 1.0;
    for(int i = 0; i < dimension; i++)
    {
        const double x_i = input[i];
        
        sum_term += pow(x_i, 2) / griewangk_constant;
        product_term *= cos(x_i / sqrt(i + 1.0));
    }
    
    const double ans = 1.0 + sum_term - product_term;
    
    return ans;
}

double MathEquations::SineEnvelopeSineWave(double input[dimension])
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

double MathEquations::StretchVSineWave(double input[dimension])
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

double MathEquations::AckleyOne(double input[dimension])
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

double MathEquations::AckelyTwo(double input[dimension])
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

double MathEquations::EggHolder(double input[dimension])
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
