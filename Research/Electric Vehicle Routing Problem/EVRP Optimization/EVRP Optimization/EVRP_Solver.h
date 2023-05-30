#pragma once

#include "GraphStructure.h"

#define STR_LEN 64
#define DEBUG true
#define VERBOSE false
#define OLD_NOT_OPTIMIZED false

class EVRP_Solver
{
public:
	EVRP_Solver();
	~EVRP_Solver();

	double CalculateTotalDistance(const std::vector<int>& solution) const;
	std::vector<int> SolveEVRP();

private:
	

	double Distance(const Node& node1, const Node& node2) const;
	int FindNearestServicableNode(std::vector<bool> visited, int current, int remaining_capacity) const;
	bool AllNodesVisited(std::vector<bool> visited) const;


	int capacity;
	std::vector<Node> nodes;
	float provided_solution;
};

