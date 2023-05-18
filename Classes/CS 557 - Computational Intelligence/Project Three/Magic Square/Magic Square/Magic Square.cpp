#include <iostream>
#include <vector>
#include <algorithm>
#include <random>

using namespace std;

// Function to generate a random number between min and max (inclusive)
int getRandomNumber(int min, int max)
{
    static random_device rd;
    static mt19937 gen(rd());
    uniform_int_distribution<> dis(min, max);
    return dis(gen);
}

// Function to calculate the fitness of a magic square based on flatness
int calculateFitness(const vector<vector<int>>& square)
{
    int n = square.size();
    int targetSum = n * (n * n + 1) / 2;
    int fitness = 0;

    // Calculate the flatness based on row sums
    for (int i = 0; i < n; i++)
    {
        int rowSum = 0;
        for (int j = 0; j < n; j++)
        {
            rowSum += square[i][j];
        }
        fitness += abs(rowSum - targetSum);
    }

    // Calculate the flatness based on column sums
    for (int j = 0; j < n; j++)
    {
        int colSum = 0;
        for (int i = 0; i < n; i++)
        {
            colSum += square[i][j];
        }
        fitness += abs(colSum - targetSum);
    }

    // Calculate the flatness based on diagonal sums
    int diagSum1 = 0;
    int diagSum2 = 0;
    for (int i = 0; i < n; i++)
    {
        diagSum1 += square[i][i];
        diagSum2 += square[i][n - i - 1];
    }
    fitness += abs(diagSum1 - targetSum);
    fitness += abs(diagSum2 - targetSum);

    return fitness;
}

// Function to generate a random magic square
vector<vector<int>> generateRandomSquare(int n)
{
    vector<int> numbers(n * n);
    for (int i = 0; i < n * n; i++)
    {
        numbers[i] = i + 1;
    }
    random_shuffle(numbers.begin(), numbers.end());

    vector<vector<int>> square(n, vector<int>(n));
    int index = 0;
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            square[i][j] = numbers[index++];
        }
    }

    return square;
}

// Function to mutate a magic square by swapping two elements
void mutateSquare(vector<vector<int>>& square)
{
    int n = square.size();
    int mutationCount = n * n / 4; // Adjust the mutation count based on the square size
    while (mutationCount > 0)
    {
        int i1 = getRandomNumber(0, n - 1);
        int j1 = getRandomNumber(0, n - 1);
        int i2 = getRandomNumber(0, n - 1);
        int j2 = getRandomNumber(0, n - 1);
        if (square[i1][j1] != square[i2][j2])
        { // Avoid unnecessary swaps
            swap(square[i1][j1], square[i2][j2]);
            mutationCount--;
        }
    }
}

// Function to perform partially mapped crossover (PMX)
vector<vector<int>> crossover(const vector<vector<int>>& parent1, const vector<vector<int>>& parent2)
{
    int n = parent1.size();
    vector<vector<int>> child(n, vector<int>(n));

    // Select two random column indices
    int colIndex1 = getRandomNumber(0, n - 1);
    int colIndex2 = getRandomNumber(0, n - 1);

    // Make sure the second column index is different from the first
    while (colIndex2 == colIndex1)
    {
        colIndex2 = getRandomNumber(0, n - 1);
    }

    // Swap the column indices if necessary to ensure colIndex1 < colIndex2
    if (colIndex1 > colIndex2)
    {
        swap(colIndex1, colIndex2);
    }

    // Copy the elements between the selected columns from parent1 to the child
    for (int i = 0; i < n; i++)
    {
        for (int j = colIndex1; j <= colIndex2; j++)
        {
            child[i][j] = parent1[i][j];
        }
    }

    // Map the elements from parent2 to the child while preserving uniqueness
    for (int i = 0; i < n; i++)
    {
        if (i >= colIndex1 && i <= colIndex2)
        {
            continue; // Skip the selected columns
        }

        // Find the elements from parent2 that are not in the child
        vector<int> missingElements;
        for (int j = 0; j < n; j++)
        {
            if (find(child[i].begin(), child[i].end(), parent2[i][j]) == child[i].end())
            {
                missingElements.push_back(parent2[i][j]);
            }
        }

        // Map the missing elements from parent2 to the corresponding positions in the child
        for (int j = 0; j < n; j++)
        {
            if (child[i][j] == 0 && !missingElements.empty())
            {
                child[i][j] = missingElements.front();
                missingElements.erase(missingElements.begin());
            }
        }
    }

    return child;
}



// Function to perform genetic algorithm to find a magic square
vector<vector<int>> findMagicSquare(int n, int populationSize, int maxIterations, int& numIterations)
{
    vector<vector<int>> bestSquare;
    int bestFitness = numeric_limits<int>::max();

    // Generate initial population
    vector<vector<vector<int>>> population(populationSize);
    for (int i = 0; i < populationSize; i++)
    {
        population[i] = generateRandomSquare(n);
    }

    // Perform evolution
    for (int iteration = 0; iteration < maxIterations; iteration++)
    {
        numIterations++;
        if (numIterations % 100 == 0)
        {
            cout << "On iteration " << numIterations << endl;
        }

        // Calculate fitness for each individual
        vector<int> fitnesses(populationSize);
        for (int i = 0; i < populationSize; i++)
        {
            fitnesses[i] = calculateFitness(population[i]);
        }

        // Find the best individual
        int minFitness = *min_element(fitnesses.begin(), fitnesses.end());
        int bestIndex = distance(fitnesses.begin(), min_element(fitnesses.begin(), fitnesses.end()));

        // Check if we found a perfect solution
        if (minFitness == 0)
        {
            bestSquare = population[bestIndex];
            cout << "Found perfect solution, breaking early" << endl;
            break;
        }

        // Update the best solution if necessary
        if (minFitness < bestFitness)
        {
            bestFitness = minFitness;
            bestSquare = population[bestIndex];
            cout << "Found new solution with fitness: " << bestFitness << endl;
        }

        // Create a new population using tournament selection, crossover, and mutation
        vector<vector<vector<int>>> newPopulation(populationSize);
        for (int i = 0; i < populationSize; i++)
        {
            // Tournament selection: select two random individuals and choose the fittest one as parent 1
            int parentIndex1 = getRandomNumber(0, populationSize - 1);
            int parentIndex2 = getRandomNumber(0, populationSize - 1);
            vector<vector<int>> parent1 = population[parentIndex1];
            vector<vector<int>> parent2 = population[parentIndex2];
            if (fitnesses[parentIndex1] > fitnesses[parentIndex2])
            {
                swap(parent1, parent2);
            }

            vector<vector<int>> child = crossover(parent1, parent2);
            mutateSquare(child);
            newPopulation[i] = child;
        }

        population = newPopulation;
    }

    return bestSquare;
}

// Function to print a magic square
void printSquare(const vector<vector<int>>& square)
{
    for (const auto& row : square)
    {
        for (int num : row)
        {
            cout << num << "\t";
        }
        cout << endl;
    }
}

void printSums(const vector<vector<int>>& square)
{
    int n = square.size();

    // Calculate the flatness based on row sums
    for (int i = 0; i < n; i++)
    {
        int rowSum = 0;
        for (int j = 0; j < n; j++)
        {
            rowSum += square[i][j];
        }
        cout << "Row " << i + 1 << " sum  = " << rowSum << endl;
    }

    // Calculate the flatness based on column sums
    for (int j = 0; j < n; j++)
    {
        int colSum = 0;
        for (int i = 0; i < n; i++)
        {
            colSum += square[i][j];
        }
        cout << "Col " << j + 1 << " sum  = " << colSum << endl;
    }

    // Calculate the flatness based on diagonal sums
    int diagSum1 = 0;
    int diagSum2 = 0;
    for (int i = 0; i < n; i++)
    {
        diagSum1 += square[i][i];
        diagSum2 += square[i][n - i - 1];
    }
    cout << "Diag 1 sum = " << diagSum1 << endl;
    cout << "Diag 2 sum = " << diagSum2 << endl;
}

int main()
{
    int n;
    cout << "Enter the order of the magic square (1 <= n <= 10): ";
    cin >> n;

    if (n > 10 || n < 1)
    {
        cout << "Invalid input. n should be between 1 and 10." << endl;
        return 0;
    }

    int populationSize = 5000;
    int maxIterations = 1000;
    int numIterations = 0;

    vector<vector<int>> magicSquare = findMagicSquare(n, populationSize, maxIterations, numIterations);

    cout << "Took " << numIterations << " iterations" << endl;
    cout << "Magic Square:" << endl;
    printSquare(magicSquare);
    cout << endl << "Magic sum is: " << n * (n * n + 1) / 2 << endl << "====================" <<endl;
    printSums(magicSquare);
    cout << endl << "This square has fitness level: " << calculateFitness(magicSquare) << endl;

    return 0;
}
