#include "GeneticAlgorithmOptimizer.h"

/*
std::vector<int> GeneticAlgorithmOptimizer::GeneticAlgorithm(const std::vector<Node>& nodes)
{
    //Initialize population
    std::vector<std::vector<int>> population;
    for (int i = 0; i < POPULATION_SIZE; i++)
    {
        population.push_back(GenerateRandomTour(nodes.size()));
    }

    //Evolve population
    for (int generation = 0; generation < MAX_GENERATIONS; generation++)
    {
        if (generation % 100 == 0)
        {
            std::cout << "Generation: " << generation << std::endl;
        }
        std::vector<std::vector<int>> newPopulation;

        //Perform crossover and mutation
        for (int i = 0; i < POPULATION_SIZE; i++)
        {
            std::vector<int> parent1 = TournamentSelection(population, nodes);
            std::vector<int> parent2 = TournamentSelection(population, nodes);
            std::vector<int> child = Crossover(parent1, parent2);
            if (std::rand() < MUTATION_RATE)
            {
                Mutate(child);
            }
            newPopulation.push_back(child);
        }
        population = newPopulation;
    }

    //Find the best tour in the final population
    std::vector<int> bestTour;
    double bestDistance = std::numeric_limits<double>::max();
    for (const auto& tour : population)
    {
        double tourDistance = CalculateTourDistance(tour, nodes);
        if (tourDistance < bestDistance)
        {
            bestTour = tour;
            bestDistance = tourDistance;
        }
    }

    return bestTour;
}

double GeneticAlgorithmOptimizer::Distance(const Node& node1, const Node& node2)
{
    int dx = node1.x - node2.x;
    int dy = node1.y - node2.y;
    return std::sqrt(std::pow(dx, 2) + std::pow(dx, 2));
    //return std::hypot(dx, dy); //guaranteed to never underflow or overflow, but much slower
}

double GeneticAlgorithmOptimizer::CalculateTourDistance(const std::vector<int>& tour, const std::vector<Node>& nodes)
{
    double dist = 0.0;
    for (int i = 0; i < tour.size() - 1; i++)
    {
        dist += Distance(nodes[tour[i]], nodes[tour[i + 1]]);
    }
    return dist;
}

std::vector<int> GeneticAlgorithmOptimizer::GenerateRandomTour(int size)
{
    std::vector<int> tour(size);
    for (int i = 0; i < size; i++)
    {
        tour[i] = i;
    }
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(tour.begin(), tour.end(), rng);
    return tour;
}

std::vector<int> GeneticAlgorithmOptimizer::Crossover(const std::vector<int>& parent1, const std::vector<int>& parent2)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> distribution(0, parent1.size() - 1);
    int start = distribution(rng);
    int end = distribution(rng);

    std::vector<int> child(parent1.size(), -1);
    for (int i = start; i <= end; i++)
    {
        child[i] = parent1[i];
    }

    int parent2Index = 0;
    for (int i = 0; i < parent2.size(); i++)
    {
        if (std::find(child.begin(), child.end(), parent2[i]) == child.end())
        {
            while (child[parent2Index] != -1)
            {
                ++parent2Index;
            }
            child[parent2Index] = parent2[i];
        }
    }

    return child;
}

void GeneticAlgorithmOptimizer::Mutate(std::vector<int>& tour)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> distribution(0, tour.size() - 1);
    int i1 = distribution(rng);
    int i2 = distribution(rng);
    std::swap(tour[i1], tour[i2]);
}

std::vector<int> GeneticAlgorithmOptimizer::TournamentSelection(const std::vector<std::vector<int>>& population, const std::vector<Node>& nodes)
{
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> distribution(0, population.size() - 1);

    std::vector<int> bestTour;
    double bestDistance = std::numeric_limits<double>::max();

    for (int i = 0; i < TOURNAMENT_SIZE; ++i)
    {
        int randomIndex = distribution(rng);
        std::vector<int> tour = population[randomIndex];
        double tourDistance = CalculateTourDistance(tour, nodes);
        if (tourDistance < bestDistance)
        {
            bestTour = tour;
            bestDistance = tourDistance;
        }
    }

    return bestTour;
}
*/