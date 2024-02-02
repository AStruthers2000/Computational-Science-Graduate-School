#pragma once
#include "../BenchmarkEquation.h"

class RosenbrockSaddle : public BenchmarkEquation
{
public:
    RosenbrockSaddle(const int& lower, const int& upper, const string& name)
        : BenchmarkEquation(lower, upper, name)
    {
    }

    double Evaluate(const vector<double>& input) override;
};
