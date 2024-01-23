#include <iostream>
#include "Math/MathEquations.h"

using namespace std;
int main(int argc, char* argv[])
{
    auto eq = new MathEquations();
    
    cout << "The equations are of dimension " << dimension << endl;
    cout << "The ackley one constant is: " << ackleys_one_constant << endl;
    cout << "The ackley two constant 1 is: " << ackelys_two_e_02 << endl;
    cout << "The ackley two constant 2 is: " << ackelys_two_e_05 << endl;
    cout << "Two pi = " << PI_2 << endl;

    double solution_vector[dimension];
    for(int i = 0; i < dimension; i++)
    {
        solution_vector[i] = 0;
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

    return 0;
}
