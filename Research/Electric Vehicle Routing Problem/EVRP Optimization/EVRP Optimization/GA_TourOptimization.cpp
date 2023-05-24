#include "GA_TourOptimization.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>

GA_TourOptimization::GA_TourOptimization(const std::vector<Node>& nodes, int capacity, int populationSize, int maxGenerations, double mutationRate)
    : graph(nodes), capacity(capacity), populationSize(populationSize), maxGenerations(maxGenerations), mutationRate(mutationRate)
{
}

std::vector<Node> GA_TourOptimization::optimize()
{
    initializePopulation();

    for (int generation = 0; generation < maxGenerations; ++generation)
    {
        for (auto& individual : population)
        {
            individual.fitness = evaluateFitness(individual);
        }

        Individual bestIndividual = findBestIndividual(population);

        std::cout << "Generation: " << generation + 1 << "\tBest Fitness: " << bestIndividual.fitness << std::endl;

        evolvePopulation();
    }

    Individual bestIndividual = findBestIndividual(population);
    return decodeChromosome(bestIndividual.chromosome);
}

int GA_TourOptimization::randomInt(int min, int max)
{
    static std::random_device rd;
    static std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(min, max);
    return dist(rng);
}

void GA_TourOptimization::initializePopulation()
{
    population.resize(populationSize);

    for (int i = 0; i < populationSize; ++i)
    {
        std::vector<int> chromosome(graph.size());
        std::iota(chromosome.begin(), chromosome.end(), 0);
        std::random_shuffle(chromosome.begin(), chromosome.end());

        population[i].chromosome = chromosome;
        population[i].fitness = 0.0;
    }
}

double GA_TourOptimization::evaluateFitness(const Individual& individual)
{
    int subtours = 0;
    int totalDemand = 0;
    int currentCapacity = 0;

    for (int i = 0; i < graph.size(); ++i)
    {
        int nodeIndex = individual.chromosome[i];
        int nodeDemand = graph[nodeIndex].demand;

        if (currentCapacity + nodeDemand > capacity)
        {
            subtours++;
            currentCapacity = nodeDemand;
        }
        else
        {
            currentCapacity += nodeDemand;
        }

        totalDemand += nodeDemand;
    }

    double fitness = 1.0 / (subtours + 1);
    if (totalDemand > capacity)
    {
        fitness *= capacity / static_cast<double>(totalDemand);
    }

    return fitness;
}

GA_TourOptimization::Individual GA_TourOptimization::crossover(const Individual& parent1, const Individual& parent2)
{
    int size = parent1.chromosome.size();
    int point1 = randomInt(0, size - 1);
    int point2 = randomInt(0, size - 1);

    int minPoint = std::min(point1, point2);
    int maxPoint = std::max(point1, point2);

    std::vector<int> offspringChromosome(size, -1);

    for (int i = minPoint; i <= maxPoint; ++i)
    {
        offspringChromosome[i] = parent1.chromosome[i];
    }

    int index = 0;
    for (int i = 0; i < size; ++i)
    {
        if (offspringChromosome[i] == -1)
        {
            while (std::find(offspringChromosome.begin(), offspringChromosome.end(), parent2.chromosome[index]) != offspringChromosome.end())
            {
                index++;
            }
            offspringChromosome[i] = parent2.chromosome[index];
        }
    }

    Individual offspring;
    offspring.chromosome = offspringChromosome;
    offspring.fitness = 0.0;

    return offspring;
}

void GA_TourOptimization::mutate(Individual& individual)
{
    for (int i = 0; i < individual.chromosome.size(); ++i)
    {
        if (randomInt(0, 100) / 100.0 < mutationRate)
        {
            int j = randomInt(0, individual.chromosome.size() - 1);
            std::swap(individual.chromosome[i], individual.chromosome[j]);
        }
    }
}

void GA_TourOptimization::evolvePopulation()
{
    std::vector<Individual> matingPool = selection();
    std::vector<Individual> newPopulation;

    for (int i = 0; i < populationSize; ++i)
    {
        const Individual& parent1 = matingPool[randomInt(0, populationSize - 1)];
        const Individual& parent2 = matingPool[randomInt(0, populationSize - 1)];

        Individual offspring = crossover(parent1, parent2);
        mutate(offspring);

        offspring.fitness = evaluateFitness(offspring);
        newPopulation.push_back(offspring);
    }

    population = newPopulation;
}

std::vector<GA_TourOptimization::Individual> GA_TourOptimization::selection()
{
    std::vector<Individual> matingPool;

    for (int i = 0; i < populationSize; ++i)
    {
        int index1 = randomInt(0, populationSize - 1);
        int index2 = randomInt(0, populationSize - 1);

        const Individual& individual1 = population[index1];
        const Individual& individual2 = population[index2];

        if (individual1.fitness > individual2.fitness)
        {
            matingPool.push_back(individual1);
        }
        else
        {
            matingPool.push_back(individual2);
        }
    }

    return matingPool;
}

GA_TourOptimization::Individual GA_TourOptimization::findBestIndividual(const std::vector<Individual>& population)
{
    auto bestIndividual = population.front();

    for (const auto& individual : population)
    {
        if (individual.fitness > bestIndividual.fitness)
        {
            bestIndividual = individual;
        }
    }

    return bestIndividual;
}

std::vector<Node> GA_TourOptimization::decodeChromosome(const std::vector<int>& chromosome)
{
    std::vector<Node> decodedGraph(graph.size());

    for (int i = 0; i < chromosome.size(); ++i)
    {
        decodedGraph[i] = graph[chromosome[i]];
    }

    return decodedGraph;
}
