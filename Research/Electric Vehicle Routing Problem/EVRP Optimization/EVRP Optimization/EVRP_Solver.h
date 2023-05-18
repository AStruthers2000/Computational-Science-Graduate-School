#pragma once
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <limits>

#define STR_LEN 64
#define DEBUG false
#define VERBOSE false

using namespace std;

class EVRP_Solver
{
public:
	EVRP_Solver();
	~EVRP_Solver();

	double CalculateTotalDistance(const vector<int>& solution) const;
	vector<int> SolveEVRP();

private:
	typedef struct
	{
		double x;
		double y;
		int demand;
	} Node;

	double Distance(const Node& node1, const Node& node2) const;
	int FindNearestServicableNode(vector<bool> visited, int current, int remaining_capacity) const;
	bool AllNodesVisited(vector<bool> visited) const;


	int capacity;
	vector<Node> nodes;
	float provided_solution;
};

