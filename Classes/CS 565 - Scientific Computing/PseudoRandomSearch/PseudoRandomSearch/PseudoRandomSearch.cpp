#include <iostream>
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
int main(int argc, char* argv[])
{
    vector<shared_ptr<BenchmarkEquation>> algorithms =
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

    for(const auto& alg : algorithms)
    {
        const double input[dimension] = {1.0};
        //double val = alg->Evaluate(input);
        cout << alg->GetName() << endl;
    }
    /*
    auto eq = new MathEquations();
    
    cout << "The equations are of dimension " << dimension << endl;
    cout << "The ackley one constant is: " << ackleys_one_constant << endl;
    cout << "The ackley two constant 1 is: " << ackelys_two_e_02 << endl;
    cout << "The ackley two constant 2 is: " << ackelys_two_e_05 << endl;
    cout << "Two pi = " << PI_2 << endl;

    double solution_vector[dimension];
    for (double& i : solution_vector)
    {
        i = rand() % (envelope_range * 2) - envelope_range;
    }

    cout << "Schwefel:                    " << eq->Schwefel(solution_vector) << endl;
    cout << "Min Schwefel:                " << min_schwefel << endl;
    cout << "=====================================" << endl;

    cout << "DeJongs1:                    " << eq->DeJong1(solution_vector) << endl;
    cout << "Min DeJongs1:                " << min_dejong1 << endl;
    cout << "=====================================" << endl;

    cout << "Rosenbrock Saddle:           " << eq->RosenbrockSaddle(solution_vector) << endl;
    cout << "Min Rosenbrock Saddle:       " << min_rosenbrocksaddle << endl;
    cout << "=====================================" << endl;

    cout << "Rastrigin:                   " << eq->Rastrigin(solution_vector) << endl;
    cout << "Min Rastrigin:               " << min_rastrigin << endl;
    cout << "=====================================" << endl;

    cout << "Griewangk:                   " << eq->Griewangk(solution_vector) << endl;
    cout << "Min Griewangk:               " << min_griewangk << endl;
    cout << "=====================================" << endl;

    cout << "Sine Envelope Sine Wave:     " << eq->SineEnvelopeSineWave(solution_vector) << endl;
    cout << "Min Sine Envelope Sine Wave: " << min_envelope << endl;
    cout << "=====================================" << endl;

    cout << "Stretch V Sine Wave:         " << eq->StretchVSineWave(solution_vector) << endl;
    cout << "Min Stretch V Sine Wave:     " << min_stretchvsine << endl;
    cout << "=====================================" << endl;

    cout << "Ackley One:                  " << eq->AckleyOne(solution_vector) << endl;
    cout << "Min Ackley One:              " << min_ackleyone << endl;
    cout << "=====================================" << endl;

    cout << "Ackley Two:                  " << eq->AckelyTwo(solution_vector) << endl;
    cout << "Min Ackley Two:              " << min_ackleytwo << endl;
    cout << "=====================================" << endl;

    cout << "Egg Holder:                  " << eq->EggHolder(solution_vector) << endl;
    cout << "Min Egg Holder:              " << min_eggholder << endl;
    cout << "=====================================" << endl;
    */
    return 0;
}
