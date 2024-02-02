#pragma once
#include "../BenchmarkEquation.h"

class Rastrigin : public BenchmarkEquation
{
public:
    Rastrigin(const int& lower, const int& upper, const string& name)
        : BenchmarkEquation(lower, upper, name)
    {
    }

    double Evaluate(const vector<double>& input) override;
};
