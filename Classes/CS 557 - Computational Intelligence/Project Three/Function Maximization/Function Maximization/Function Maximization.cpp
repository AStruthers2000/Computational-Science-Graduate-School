/*
@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
*/

#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#define _USE_MATH_DEFINES
#include <math.h>
#include <fstream>
//#include "Testing.h"


// Function to maximize (example: f(x, y) = x^2 + y^2)
double f(double x, double y) {
    //return x * x + y * y;
    return sin(M_PI * 10 * x + (10.0 / (1 + (y * y)))) + log(x * x + y * y);  
}

// Genetic Algorithm parameters
const int NUM_GENERATIONS = 1000;
const int POPULATION_SIZE = 100000;
const double MUTATION_PROBABILITY = 0.05;

// Bounds of x and y
const double X_LOWER_BOUND = 3;
const double X_UPPER_BOUND = 10;
const double Y_LOWER_BOUND = 4;
const double Y_UPPER_BOUND = 8;

// Random number generator
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<double> dis_x(X_LOWER_BOUND, X_UPPER_BOUND);
std::uniform_real_distribution<double> dis_y(Y_LOWER_BOUND, Y_UPPER_BOUND);
std::uniform_real_distribution<double> dis_mutation(0, 1);

// Individual representation
struct Individual
{
    double x;
    double y;
    double fitness;
};

// Function to evaluate the fitness of an individual
void evaluateFitness(Individual& individual)
{
    individual.fitness = f(individual.x, individual.y);
}

// Function to create an initial population
std::vector<Individual> createInitialPopulation()
{
    std::vector<Individual> population;
    for (int i = 0; i < POPULATION_SIZE; ++i)
    {
        population.push_back({ dis_x(gen), dis_y(gen), 0 });
        evaluateFitness(population[i]);
    }
    return population;
}

// Function to perform selection
Individual selection(const std::vector<Individual>& population)
{
    std::uniform_int_distribution<int> dis_selection(0, POPULATION_SIZE - 1);
    int index1 = dis_selection(gen);
    int index2 = dis_selection(gen);
    return (population[index1].fitness > population[index2].fitness) ? population[index1] : population[index2];
}

// Function to perform crossover
Individual crossover(const Individual& parent1, const Individual& parent2)
{
    return { (parent1.x + parent2.x) / 2, (parent1.y + parent2.y) / 2, 0 };
}

// Function to perform mutation
void mutate(Individual& individual)
{
    if (dis_mutation(gen) < MUTATION_PROBABILITY)
    {
        individual.x = dis_x(gen);
        individual.y = dis_y(gen);
    }
}

// Function to evolve the population for one generation
void evolvePopulation(std::vector<Individual>& population)
{
    std::vector<Individual> newPopulation;
    for (int i = 0; i < POPULATION_SIZE; ++i)
    {
        // Selection
        Individual parent1 = selection(population);
        Individual parent2 = selection(population);

        // Crossover
        Individual child = crossover(parent1, parent2);

        // Mutation
        mutate(child);

        evaluateFitness(child);
        newPopulation.push_back(child);
    }
    population = newPopulation;
}

// Function to find the best individual in a population
Individual findBestIndividual(const std::vector<Individual>& population)
{
    auto maxFitnessIt = std::max_element(population.begin(), population.end(),
        [](const Individual& a, const Individual& b)
        {
            return a.fitness < b.fitness;
        });
    return *maxFitnessIt;
}

int main()
{
    /*
    Test* t = new Test();
    float max_x = 0, max_y = 0;

    float func_max = t->calc_max(max_x, max_y);
    std::cout << "max(f(x, y)) = " << func_max << " at (" << max_x << ", " << max_y << ")" << std::endl;
    */
    // Create initial population
    std::vector<Individual> population = createInitialPopulation();

    // Evolution of average fitness value
    std::vector<double> avgFitnessEvolution;

    // Genetic Algorithm main loop
    for (int generation = 0; generation < NUM_GENERATIONS; ++generation)
    {
        // Calculate average fitness
        double sumFitness = 0;
        for (const auto& individual : population)
        {
            sumFitness += individual.fitness;
        }
        double avgFitness = sumFitness / POPULATION_SIZE;
        avgFitnessEvolution.push_back(avgFitness);

        // Find the best individual
        Individual bestIndividual = findBestIndividual(population);

        // Print current generation information
        std::cout << "Generation " << (generation + 1) << ":\n";
        std::cout << "Best Individual: (" << bestIndividual.x << ", " << bestIndividual.y << ")\n";
        std::cout << "Best Fitness: " << bestIndividual.fitness << "\n";
        std::cout << "Average Fitness: " << avgFitness << "\n\n";

        // Evolve the population
        evolvePopulation(population);
    }

    std::cout << "Number of Generations: " << NUM_GENERATIONS << "\n";
    std::cout << "Population Size: " << POPULATION_SIZE << "\n";
    std::cout << "Mutation Probability: " << MUTATION_PROBABILITY << "\n";

    std::ofstream file("average_fitness.txt");
    std::ostream_iterator<double> o_iter(file, "\n");
    std::copy(std::begin(avgFitnessEvolution), std::end(avgFitnessEvolution), o_iter);
    file.close();
    
    return 0;
}
