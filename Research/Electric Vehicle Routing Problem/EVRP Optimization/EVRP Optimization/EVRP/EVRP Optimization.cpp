// EVRP Optimization.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include "EVRP_Solver.h"

int main()
{
    EVRP_Solver* solver = new EVRP_Solver();
    std::vector<int> solution = solver->SolveEVRP();
    /*
    std::cout << "Optimal tour: ";
    for (int i = 0; i < solution.size(); i++)
    {
        std::cout << solution[i] << " ";
    }
    std::cout << std::endl;
    */

    /*
    double objective = solver->CalculateTotalDistance(solution);
    cout << "Objective: " << objective << endl;
    cout << "Route: " << endl;
    for (int node : solution)
    {
        cout << node << " ";
    }
    */

    return 0;
}