#pragma once
#include "../BenchmarkEquation.h"

class AckleyTwo : public BenchmarkEquation
{
public:
    AckleyTwo(const int& lower, const int& upper, const string& name)
        : BenchmarkEquation(lower, upper, name)
    {
    }

    double Evaluate(const double input[dimension]) override;
};
