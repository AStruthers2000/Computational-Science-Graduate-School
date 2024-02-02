#include "Griewangk.h"

double Griewangk::Evaluate(const vector<double>& input)
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
