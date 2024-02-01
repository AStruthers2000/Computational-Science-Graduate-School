#pragma once
#include "../BenchmarkEquation.h"

class Griewangk : public BenchmarkEquation
{
public:
    Griewangk(const int& lower, const int& upper, const string& name)
        : BenchmarkEquation(lower, upper, name)
    {
    }

    double Evaluate(const double input[dimension]) override;
};
