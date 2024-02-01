#pragma once
#include "../BenchmarkEquation.h"

class SineEnvelopeSineWave : public BenchmarkEquation
{
public:
    SineEnvelopeSineWave(const int& lower, const int& upper, const string& name)
        : BenchmarkEquation(lower, upper, name)
    {
    }

    double Evaluate(const double input[dimension]) override;
};
