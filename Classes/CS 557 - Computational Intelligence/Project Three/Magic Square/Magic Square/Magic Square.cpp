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
    int i1 = getRandomNumber(0, n - 1);
    int j1 = getRandomNumber(0, n - 1);
    int i2 = getRandomNumber(0, n - 1);
    int j2 = getRandomNumber(0, n - 1);
    swap(square[i1][j1], square[i2][j2]);
}

// Function to perform genetic algorithm to find a magic square
vector<vector<int>> findMagicSquare(int n, int populationSize, int maxIterations, int &numIterations) 
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
            cout << "Current iteration: " << numIterations << endl;
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
            cout << "Perfect solution found, ending early" << endl;
            break;
        }

        // Update the best solution if necessary
        if (minFitness < bestFitness) 
        {
            bestFitness = minFitness;
            bestSquare = population[bestIndex];
            cout << "Found new best solution with fitness: " << bestFitness << endl;
        }

        // Create a new population using tournament selection and mutation
        vector<vector<vector<int>>> newPopulation(populationSize);
        for (int i = 0; i < populationSize; i++) 
        {
            int parentIndex1 = getRandomNumber(0, populationSize - 1);
            int parentIndex2 = getRandomNumber(0, populationSize - 1);
            newPopulation[i] = population[parentIndex1];
            if (fitnesses[parentIndex2] < fitnesses[parentIndex1]) 
            {
                newPopulation[i] = population[parentIndex2];
            }
            mutateSquare(newPopulation[i]);
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
    constexpr int maxIterations = 2000;//numeric_limits<int>::max();
    int numIterations = 0;

    vector<vector<int>> magicSquare = findMagicSquare(n, populationSize, maxIterations, numIterations);
    cout << "Took " << numIterations << " iterations" << endl;
    cout << "Magic Square:" << endl;
    printSquare(magicSquare);
    cout << endl << "Magic sum is: " << n * (n * n + 1) / 2 << endl;
    printSums(magicSquare);
    cout << endl << "This square has fitness level: " << calculateFitness(magicSquare) << endl;
    return 0;
}
