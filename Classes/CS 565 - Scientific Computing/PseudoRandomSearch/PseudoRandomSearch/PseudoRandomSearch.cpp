#include <cassert>
#include <iostream>
#include <random>
#include <vector>

#include "Math/BenchmarkEquation.h"
#include "Math/MathEquations.h"
#include "Math/Equations/AckleyOne.h"
#include "Math/Equations/AckleyTwo.h"
#include "Math/Equations/DeJong1.h"
#include "Math/Equations/EggHolder.h"
#include "Math/Equations/Griewangk.h"
#include "Math/Equations/Rastrigin.h"
#include "Math/Equations/RosenbrockSaddle.h"
#include "Math/Equations/Schwefel.h"
#include "Math/Equations/SineEnvelopeSineWave.h"
#include "Math/Equations/StretchVSineWave.h"

using namespace std;

class PRNG_Optimizer
{
    
public:
    vector<double> GenerateRandomInput_MersenneTwister(const shared_ptr<BenchmarkEquation>& eq) const;
    vector<double> GenerateRandomInput_LaggedFibonacci(const shared_ptr<BenchmarkEquation>& eq);

    void Init_LFG_State(const shared_ptr<BenchmarkEquation>& eq);
    
private:
    double MersenneTwister(int inclusive_lower, int inclusive_upper) const;
    double LaggedFibonacci();
    
    vector<double> LFG_state_buffer;
    int LFG_range = 0;

    enum LFG_state
    {
        LFG_j = 97,
        LFG_k = 127,
    };
};

vector<double> PRNG_Optimizer::GenerateRandomInput_MersenneTwister(const shared_ptr<BenchmarkEquation>& eq) const
{
    vector<double> input_vector;
    input_vector.reserve(dimension);
    for (int i = 0; i < dimension; i++)
    {
        input_vector.push_back(MersenneTwister(eq->GetLowerBound(), eq->GetUpperBound()));
        assert(i >= eq->GetLowerBound() && i <= eq->GetUpperBound());
    }
    return input_vector;
}

vector<double> PRNG_Optimizer::GenerateRandomInput_LaggedFibonacci(const shared_ptr<BenchmarkEquation>& eq)
{
    assert(LFG_range != 0);
    vector<double> input_vector;
    input_vector.reserve(dimension);
    for (int i = 0; i < dimension; i++)
    {
        input_vector.push_back(LaggedFibonacci());
        assert(i >= eq->GetLowerBound() && i <= eq->GetUpperBound());
    }
    return input_vector;
}

void PRNG_Optimizer::Init_LFG_State(const shared_ptr<BenchmarkEquation>& eq)
{
    LFG_state_buffer.clear();
    LFG_range = abs(eq->GetLowerBound()) + abs(eq->GetUpperBound());
    for(int i = 0; i < LFG_k; i++)
    {
        const double r = MersenneTwister(0, LFG_range);
        LFG_state_buffer.push_back(fmod(r, LFG_range));
    }
}

double PRNG_Optimizer::MersenneTwister(int inclusive_lower, int inclusive_upper) const
{
    random_device rd;
    mt19937_64 mt(rd());

    uniform_real_distribution<> distribution(inclusive_lower, inclusive_upper);
    const double r = distribution(mt);
    return r;
}

double PRNG_Optimizer::LaggedFibonacci()
{
    const double bin_op = LFG_state_buffer[LFG_j - 1] + LFG_state_buffer[LFG_k - 1];
    const double next = fmod(bin_op, LFG_range); //- LFG_range / 2.0;
    LFG_state_buffer.push_back(next);
    LFG_state_buffer.erase(LFG_state_buffer.begin());
    return next - LFG_range / 2.0;
}

constexpr int experiments = 30;
int main(int argc, char* argv[])
{
    const vector<shared_ptr<BenchmarkEquation>> algorithms =
    {
        make_shared<Schwefel>(-512, 512, "Schwefel"),
        make_shared<DeJong1>(-100, 100, "De Jong's 1"),
        make_shared<RosenbrockSaddle>(-100, 100, "Rosenbrock's Saddle"),
        make_shared<Rastrigin>(-30, 30, "Rastrigin"),
        make_shared<Griewangk>(-500, 500, "Griewangk"),
        make_shared<SineEnvelopeSineWave>(-30, 30, "Sine Envelope Sine Wave"),
        make_shared<StretchVSineWave>(-30, 30, "Stretch V Sine Wave"),
        make_shared<AckleyOne>(-32, 32, "Ackley's One"),
        make_shared<AckleyTwo>(-32, 32, "Ackley's Two"),
        make_shared<EggHolder>(-500, 500, "Egg Holder")
    };

    const auto optimizer = make_shared<PRNG_Optimizer>();
    
    for(const auto& alg : algorithms)
    {
        vector<double> results_mt;
        vector<double> results_lfg;
        vector<double> results_bbs;

        double best_mt = numeric_limits<double>::max();
        double best_lfg = numeric_limits<double>::max();
        
        optimizer->Init_LFG_State(alg);
        for(int i = 0; i < experiments; i++)
        {
            const auto input_mt = optimizer->GenerateRandomInput_MersenneTwister(alg);
            const auto input_lfg = optimizer->GenerateRandomInput_LaggedFibonacci(alg);
            //double val = alg->Evaluate(input);
            
            //cout << alg->GetName() << endl;

            const double mt = alg->Evaluate(input_mt);
            if(mt < best_mt)
            {
                best_mt = mt;
            }
            
            const double lfg = alg->Evaluate(input_lfg);
            if(lfg < best_lfg)
            {
                best_lfg = lfg;
            }

            //cout << "Mersenne Twister on the " << alg->GetName() << ":           " << mt << endl;
            //cout << "Lagged Fibonacci Generator on the " << alg->GetName() << ": " << lfg << endl;
            //cout << "--------------------------------" << endl;
        }
        cout << "Stats for algorithm: " << alg->GetName() << endl;
        cout << "\tBest solution with Mersenne Twister:           " << best_mt << endl;
        cout << "\tBest solution with Lagged Fibonacci Generator: " << best_lfg << endl;
        cout << "======================================" << endl;
    }

    return 0;
}
