#pragma once
#include <string>
#include <vector>

#define PI 3.14159265358979323846
#define PI_2 (PI * 2.0)
#define E 2.71828182845904523536

constexpr double schwefel_constant = 418.9829;
constexpr double rosenbrock_constant = 100.0;
constexpr double rastrigin_constant = 10.0;
constexpr double griewangk_constant = 4000.0;
#define ackleys_one_constant (1.0 / pow(E, 0.2))
#define ackelys_two_e_02 pow(E, 0.2)
#define ackelys_two_e_05 pow(E, 0.5)
constexpr double eggholder_constant = 47.0;

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

    virtual double Evaluate(const vector<double>& input) = 0;

private:
    EvalRange _eval_range;
    string _equation_name;
};
