#pragma once
#include "../BenchmarkEquation.h"

class EggHolder : public BenchmarkEquation
{
public:
    EggHolder(const int& lower, const int& upper, const string& name)
        : BenchmarkEquation(lower, upper, name)
    {
    }

    double Evaluate(const double input[dimension]) override;
};
