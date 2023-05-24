#ifndef TOUR_OPTIMIZATION_H
#define TOUR_OPTIMIZATION_H

#include <vector>

struct Node
{
    int x;
    int y;
    int demand;
};

class GA_TourOptimization
{
public:
    GA_TourOptimization(const std::vector<Node>& nodes, int capacity, int populationSize, int maxGenerations, double mutationRate);

    std::vector<Node> optimize();

private:
    struct Individual
    {
        std::vector<int> chromosome;
        double fitness;
    };

    int randomInt(int min, int max);

    void initializePopulation();

    double evaluateFitness(const Individual& individual);

    Individual crossover(const Individual& parent1, const Individual& parent2);

    void mutate(Individual& individual);

    void evolvePopulation();

    std::vector<Individual> selection();

    Individual findBestIndividual(const std::vector<Individual>& population);

    std::vector<Node> decodeChromosome(const std::vector<int>& chromosome);

    std::vector<Node> graph;
    int capacity;
    int populationSize;
    int maxGenerations;
    double mutationRate;
    std::vector<Individual> population;
};

#endif  // GENETIC_ALGORITHM_H
