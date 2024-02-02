#pragma once
#include "../BenchmarkEquation.h"

class StretchVSineWave : public BenchmarkEquation
{
public:
    StretchVSineWave(const int& lower, const int& upper, const string& name)
        : BenchmarkEquation(lower, upper, name)
    {
    }

    double Evaluate(const vector<double>& input) override;
};
