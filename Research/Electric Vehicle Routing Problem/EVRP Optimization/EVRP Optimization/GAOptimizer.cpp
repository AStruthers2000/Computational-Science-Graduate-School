#include "GAOptimizer.h"
#include <set>

void GAOptimizer::Optimize(const std::vector<Node> nodes, const int capacity, std::vector<int> &bestTour, int &bestFitness, double &bestDistance)
{
	//std::vector<int> testRoute = { 4, 2, 0, 3, 1 };
	//printf("Test - the test route has fitness %d\n", EvaluateFitness(nodes, testRoute, capacity, true));
	//printf("Test - test route has distance: %f\n", CalculateTotalDistance(nodes, testRoute, capacity));

	std::vector<std::vector<int>> population;
	std::vector<int> fitnesses;
	std::vector<double> distances;

	//generate initial population and fitnesses
	for (int i = 0; i < POPULATION_SIZE; i++)
	{
		population.push_back(GenerateRandomTour(nodes.size()));
		int fitness;
		double distance;
		EvaluateFitness(nodes, population[i], capacity, fitness, distance);
		fitnesses.push_back(fitness);
		distances.push_back(distance);
	}

	for (int generation = 0; generation < MAX_GENERATIONS; generation++)
	{
		if ((generation+1) % 10 == 0)
		{
			system("cls");
		}
		std::cout << "Percent complete: " << (double(generation) / MAX_GENERATIONS) * 100.0 << "%" << std::endl;

		std::vector<std::vector<int>> newPopulation;
		std::vector<int> newFitnesses;
		std::vector<double> newDistances;

		for (int i = 0; i < POPULATION_SIZE; i++)
		{
			//select parents
			//perform crossover between parents
			//mutate child

			std::vector<int> parentTour1 = TournamentSelection(nodes, population, fitnesses, capacity);
			std::vector<int> parentTour2 = TournamentSelection(nodes, population, fitnesses, capacity);
			std::vector<int> childTour = Crossover(parentTour1, parentTour2);
			if (std::rand() <= MUTATION_RATE)
			{
				Mutate(childTour);
			}

			//add child to new population and calculate new fitness 
			newPopulation.push_back(childTour);

			int fitness;
			double distance;
			EvaluateFitness(nodes, childTour, capacity, fitness, distance);
			newFitnesses.push_back(fitness);
			newDistances.push_back(distance);

		}
		population = newPopulation;
		fitnesses = newFitnesses;
		distances = newDistances;
	}
	
	bestFitness = std::numeric_limits<int>::max();
	bestDistance = std::numeric_limits<double>::max();
	for (int i = 0; i < POPULATION_SIZE; i++)
	{
		std::vector<int> tour = population[i];
		int fitness = fitnesses[i];
		double distance = distances[i];

		if (fitness <= bestFitness && distance < bestDistance)
		{
			bestTour = tour;
			bestFitness = fitness;
			bestDistance = distance;
		}
	}

	std::cout << "Best tour: ";
	PrintTour(bestTour);
	std::cout << "Fitness calculations: " << std::endl;
	int numSubtours;
	double distance;
	EvaluateFitness(nodes, bestTour, capacity, numSubtours, distance, true);
}

std::vector<int> GAOptimizer::GenerateRandomTour(const int size)
{
	std::vector<int> tour(size);
	for (int i = 0; i < size; i++)
	{
		tour[i] = i;
	}
	ShuffleVector(tour);
	return tour;
}

void GAOptimizer::EvaluateFitness(const std::vector<Node> nodes, const std::vector<int> tour, const int capacity, int& numSubtours, double& distance, const bool verbose) const
{
	std::vector<int> paddedTour;
	//fitness = number of times we have to visit the depot in this tour
	//fitness should be as small as possible. it will generally be > 0
	int fitness = 0;

	paddedTour.push_back(-1);

	int current_capacity = capacity;
	for (int i = 0; i < tour.size(); i++)
	{
		int demand = nodes[tour[i]].demand;
		if(verbose)	printf("We have %d remaining supply, and node %d has demand %d\n", current_capacity, tour[i] + 1, demand);
		if (current_capacity - demand < 0)
		{
			//we needed to visit the depot befor servicing this node
			fitness++; 
			current_capacity = capacity;
			paddedTour.push_back(-1);
			if (verbose) printf("\tBefore servicing this node, we must go back to the depot. We have visited the depot %d times this tour\n", fitness);
		}
		current_capacity -= demand;
		paddedTour.push_back(tour[i]);
		if (verbose) printf("\tAfter servicing this node, we now have %d remaining supply\n", current_capacity);
	}

	//We return to the depot at the end of the tour, so we add one to the fitness
	fitness++;
	paddedTour.push_back(-1);
	if (verbose) printf("Returning to the depot at the end of the tour\n====================\n");
	if (verbose) PrintTour(paddedTour);

	distance = CalculateTotalDistance(nodes, paddedTour, capacity);
	numSubtours = fitness;
}

double GAOptimizer::CalculateTotalDistance(const std::vector<Node> nodes, const std::vector<int>& tour, const int capacity) const
{
	Node home = Node{ 0, 0, 0 };
	int current_capacity = capacity;
	double tot = 0;

	for (int i = 1; i < tour.size(); i++)
	{
		Node from;
		Node to;
		if (tour[i - 1] == -1)
		{
			from = home;
		}
		else
		{
			from = nodes[tour[i - 1]];
		}

		if (tour[i] == -1)
		{
			to = home;
		}
		else
		{
			to = nodes[tour[i]];
		}

		tot += Distance(from, to);
	}

	//printf("Total distance %f\n", tot);
	return tot;
}

double GAOptimizer::Distance(const Node& node1, const Node& node2) const
{
	double dist = hypot(node1.x - node2.x, node1.y - node2.y);
	//printf("Distance between node at (%f, %f) and (%f, %f) is %f\n", node1.x, node1.y, node2.x, node2.y, dist);
	return dist;
}

std::vector<int> GAOptimizer::TournamentSelection(const std::vector<Node> nodes, const std::vector<std::vector<int>> population, const std::vector<int> fitnesses, const int capacity) const
{
	std::vector<int> bestTour;
	int bestFitness = std::numeric_limits<int>::max();
	double bestDistance = std::numeric_limits<double>::max();

	for (int i = 0; i < TOURNAMENT_SIZE; i++)
	{
		int index = RandomNumberGenerator(0, population.size() - 1);
		std::vector<int> tour = population[index];
		int fitness = fitnesses[index];
		double distance = CalculateTotalDistance(nodes, tour, capacity);
		if (fitness <= bestFitness && distance < bestDistance)
		{
			bestTour = tour;
			bestFitness = fitness;
			bestDistance = distance;
		}
	}
	return bestTour;
}

std::vector<int> GAOptimizer::Crossover(const std::vector<int> parentTour1, const std::vector<int> parentTour2) const
{
	// Create a child vector with the same size as the parents
	std::vector<int> child(parentTour1.size());

	// Copy a random subset of elements from parent1 to the child
	int crossoverPoint = rand() % parentTour1.size();
	std::copy(parentTour1.begin(), parentTour1.begin() + crossoverPoint, child.begin());

	// Fill the remaining elements in the child with unique elements from parent2
	int childIndex = crossoverPoint;
	for (int i = 0; i < parentTour2.size(); ++i)
	{
		int element = parentTour2[i];
		// Check if the element is already present in the child
		if (std::find(child.begin(), child.end(), element) == child.end())
		{
			child[childIndex] = element;
			++childIndex;
		}
	}
	//this is to assert that the child doesn't contain any duplicates
	std::set<int> unique_s(child.begin(), child.end());
	std::vector<int> unique_v(unique_s.begin(), unique_s.end());
	for (int i = 0; i < unique_v.size()-1; i++)
	{
		if (unique_v[i] + 1 != unique_v[i + 1])
		{
			std::cout << "\n\n\nERROR IN CROSSOVER!!!!\n\n\n" << std::endl;
			PrintTour(child);
			PrintTour(unique_v);
			std::cout << "\n\n\n======================\n\n\n" << std::endl;
		}
	}

	return child;
}

void GAOptimizer::Mutate(std::vector<int>& child)
{
	int index1 = RandomNumberGenerator(0, child.size() - 1);
	int index2 = RandomNumberGenerator(0, child.size() - 1);
	std::swap(child[index1], child[index2]);
}



int GAOptimizer::RandomNumberGenerator(const int min, const int max) const
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::uniform_int_distribution<int> distr(min, max);
	return distr(generator);
}

void GAOptimizer::ShuffleVector(std::vector<int>& container)
{
	std::random_device rd;
	std::mt19937 generator(rd());
	std::shuffle(container.begin(), container.end(), generator);
}

void GAOptimizer::PrintTour(const std::vector<int> tour) const
{
	std::cout << "Tour: ";
	for (int i = 0; i < tour.size(); i++)
	{
		std::cout << tour[i] + 1 << " ";
	}
	std::cout << std::endl;
}