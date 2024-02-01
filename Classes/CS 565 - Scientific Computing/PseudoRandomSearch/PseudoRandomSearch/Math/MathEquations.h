#pragma once
#include <cmath>

/*
#define PI 3.14159265358979323846
#define PI_2 (PI * 2.0)
#define E 2.71828182845904523536
*/
//constexpr int dimension = 30;

constexpr double schwefel_constant = 418.9829;
constexpr double rosenbrock_constant = 100.0;
constexpr double rastrigin_constant = 10.0;
constexpr double griewangk_constant = 4000.0;
#define ackleys_one_constant (1.0 / pow(E, 0.2))
#define ackelys_two_e_02 pow(E, 0.2)
#define ackelys_two_e_05 pow(E, 0.5)
constexpr double eggholder_constant = 47.0;

constexpr int schwefel_range = 512;
constexpr int dejong1_range = 100;
constexpr int rosenbrock_range = 100;
constexpr int rastrigin_range = 30;
constexpr int griewangk_range = 500;
constexpr int envelope_range = 30;
constexpr int stretchvsine_range = 30;
constexpr int ackleyone_range = 32;
constexpr int ackleytwo_range = 32;
constexpr int eggholder_range = 500;

constexpr double min_schwefel = 0;
constexpr double min_dejong1 = 0;
constexpr double min_rosenbrocksaddle = 0;
constexpr double min_rastrigin = 0;
constexpr double min_griewangk = 0;
#define min_envelope (-1.4915 * (dimension - 1.0))
constexpr double min_stretchvsine = 0;
#define min_ackleyone (-7.54276 - 2.91867 * (dimension - 3))
constexpr double min_ackleytwo = 0;
constexpr double min_eggholder = 0;

class MathEquations
{
public:
    /*
    double Schwefel(double input[dimension]);
    double DeJong1(double input[dimension]);
    double RosenbrockSaddle(double input[dimension]);
    double Rastrigin(double input[dimension]);
    double Griewangk(double input[dimension]);
    double SineEnvelopeSineWave(double input[dimension]);
    double StretchVSineWave(double input[dimension]);
    double AckleyOne(double input[dimension]);
    double AckelyTwo(double input[dimension]);
    double EggHolder(double input[dimension]);
    */

private:

};
