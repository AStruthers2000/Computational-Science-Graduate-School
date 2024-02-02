#pragma once
#include "../BenchmarkEquation.h"

class AckleyOne : public BenchmarkEquation
{
public:
    AckleyOne(const int& lower, const int& upper, const string& name)
        : BenchmarkEquation(lower, upper, name)
    {
    }

    double Evaluate(const vector<double>& input) override;
};
