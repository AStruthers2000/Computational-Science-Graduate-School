#pragma once
#include "GraphStructure.h"

constexpr int POPULATION_SIZE = 200;
constexpr int MAX_GENERATIONS = 50;
constexpr int TOURNAMENT_SIZE = POPULATION_SIZE * 0.025;
constexpr float MUTATION_RATE = 0.15;

class GAOptimizer
{
public:
	void Optimize(const std::vector<Node> nodes, const int capacity, std::vector<int>& bestTour, int& bestFitness, double& bestDistance);

private:
	std::vector<int> GenerateRandomTour(const int size);
	
	void EvaluateFitness(const std::vector<Node> nodes, const std::vector<int> tour, const int capacity, int &numSubtours, double &distance, const bool verbose = false) const;
	double CalculateTotalDistance(const std::vector<Node> nodes, const std::vector<int>& tour, const int capacity) const;
	double Distance(const Node& node1, const Node& node2) const;
	
	std::vector<int> TournamentSelection(const std::vector<Node> nodes, const std::vector<std::vector<int>> population, const std::vector<int> fitnesses, const int capacity) const;
	std::vector<int> Crossover(const std::vector<int> parentTour1, const std::vector<int> parentTour2) const;
	void Mutate(std::vector<int>& child);

	int RandomNumberGenerator(const int min, const int max) const;
	void ShuffleVector(std::vector<int>& container);
	void PrintTour(const std::vector<int> tour) const;
};

