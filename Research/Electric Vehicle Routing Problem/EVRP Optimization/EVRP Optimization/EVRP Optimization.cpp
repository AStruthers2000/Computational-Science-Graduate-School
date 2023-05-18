// EVRP Optimization.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include "EVRP_Solver.h"
using namespace std;
int main()
{
    EVRP_Solver* solver = new EVRP_Solver();
    vector<int> solution = solver->SolveEVRP();
    double objective = solver->CalculateTotalDistance(solution);
    cout << "Objective: " << objective << endl;
    cout << "Route: " << endl;
    for (int node : solution)
    {
        cout << node << " ";
    }

    return 0;
}