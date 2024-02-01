#pragma once
#include <string>

#define PI 3.14159265358979323846
#define PI_2 (PI * 2.0)
#define E 2.71828182845904523536

constexpr int dimension = 30;


struct EvalRange
{
    int lower_bound = 0;
    int upper_bound = 0;
};

using namespace std;
class BenchmarkEquation
{
public:
    BenchmarkEquation(const int& lower, const int& upper, const string& name)
    {
        _eval_range = {lower, upper};
        _equation_name = name;
    }

    virtual ~BenchmarkEquation() = default;

    int GetUpperBound() const
    {
        return _eval_range.upper_bound;
    }

    int GetLowerBound() const
    {
        return _eval_range.lower_bound;
    }

    const EvalRange* GetBounds() const
    {
        return &_eval_range;
    }

    string GetName() const
    {
        return _equation_name;
    }

    virtual double Evaluate(const double input[dimension]) = 0;

private:
    EvalRange _eval_range;
    string _equation_name;
};
