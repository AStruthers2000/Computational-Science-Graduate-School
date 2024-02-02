#pragma once
#include "../BenchmarkEquation.h"

class DeJong1 : public BenchmarkEquation
{
public:
    DeJong1(const int& lower, const int& upper, const string& name)
        : BenchmarkEquation(lower, upper, name)
    {
    }

    double Evaluate(const vector<double>& input) override;
};
