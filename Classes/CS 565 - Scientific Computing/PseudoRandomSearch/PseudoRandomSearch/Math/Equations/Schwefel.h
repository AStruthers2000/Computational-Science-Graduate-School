#pragma once
#include "../BenchmarkEquation.h"

class Schwefel : public BenchmarkEquation
{
public:
    Schwefel(const int& lower, const int& upper, const string& name)
        : BenchmarkEquation(lower, upper, name)
    {
    }

    double Evaluate(const double input[dimension]) override;
};
