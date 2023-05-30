#pragma once
#include "GraphStructure.h"


//constexpr auto POPULATION_SIZE = 500;
//constexpr auto MAX_GENERATIONS = 10000;
//constexpr auto MUTATION_RATE = 0.1;
//constexpr auto TOURNAMENT_SIZE = 5;

class GeneticAlgorithmOptimizer
{
public:
	std::vector<int> GeneticAlgorithm(const std::vector<Node>& nodes);


private:
	double Distance(const Node& node1, const Node& node2);
	double CalculateTourDistance(const std::vector<int>& tour, const std::vector<Node>& nodes);
	std::vector<int> GenerateRandomTour(int size);
	std::vector<int> Crossover(const std::vector<int>& parent1, const std::vector<int>& parent2);
	void Mutate(std::vector<int>& tour);
	std::vector<int> TournamentSelection(const std::vector<std::vector<int>>& population, const std::vector<Node>& nodes);
};

