import random
import time
from datetime import datetime

TEST_DIMENSION = 200
TEST_POWER = 10

def initialize_matrix(dimension, init_type):
    matrix = []
    for i in range(dimension):
        row = []
        for j in range(dimension):
            if init_type == 'Identity':
                row.append(1 if i == j else 0)
            elif init_type == 'Random':
                row.append(random.uniform(-1.0, 1.0))
            else:
                raise ValueError("Unknown matrix initialization type")
        matrix.append(row)
    return matrix

def matrix_multiply(A, B):
    # Multiply matrix A by matrix B
    dimension = len(A)
    result = [[0.0] * dimension for _ in range(dimension)]
    for i in range(dimension):
        for j in range(dimension):
            for k in range(dimension):
                result[i][j] += A[i][k] * B[k][j]
    return result

def raise_to_power(matrix, power, verbose=False):
    current = [row[:] for row in matrix]  # Create a copy of the matrix
    for _ in range(power - 1):
        previous = [row[:] for row in current]  # Create a copy of the current matrix
        current = matrix_multiply(previous, matrix)
        if verbose:
            print(f"Raising matrix to the power of {_ + 2}")
    return current

def print_matrix(m):
    dimension = len(m)
    for i in range(dimension):
        for j in range(dimension):
            if -10000 < m[i][j] < 0:
                print(f"{m[i][j]:.4f}\t", end="")
            elif 0 <= m[i][j] < 10000:
                print(f"{m[i][j]:.5f}\t", end="")
            else:
                print(f"{m[i][j]:.5e}\t", end="")
        print()

def benchmark(matrix_dimension, matrix_power, matrix_type, verbose, filetype):

    for i in range(30):
        matrix = initialize_matrix(matrix_dimension, matrix_type)
        
        #print("Base matrix:")
        #print_matrix(matrix)
        #print(f"\n========== Starting matrix multiplication {matrix_dimension}x{matrix_dimension}^{matrix_power} ==========")
        
        start_time = time.time()
        start_wall_clock = datetime.now()

        result = raise_to_power(matrix, matrix_power, verbose)
        
        end_time = time.time()
        end_wall_clock = datetime.now()
        
        #print("\n========== Finishing matrix multiplication ==========")
        
        #print("\nFinal matrix:")
        #print_matrix(result)
        
        elapsed_time = end_time - start_time
        elapsed_wall_clock = (end_wall_clock - start_wall_clock).total_seconds()
        
        #print(f"Elapsed time using time(): \t{elapsed_time:.6f} seconds")
        #print(f"Elapsed time using datetime: \t{elapsed_wall_clock:.6f} seconds")
        print(f"{matrix_dimension},{matrix_power},{elapsed_time}")
        with open(f"python_raw_{filetype}.txt", "a") as file:
            #temp = matrix_dimension if filetype == "dimension" else matrix_power
            file.write(f"{matrix_dimension},{matrix_power},{elapsed_time}\n")
            file.close()
    

if __name__ == "__main__":

    """
    import sys
    args = sys.argv[1:]
    
    matrix_dimension = TEST_DIMENSION
    matrix_power = TEST_POWER
    matrix_type = 'Random'
    verbose = False
    
    if len(args) > 0:
        matrix_dimension = int(args[0])
        print("Provided dimension:", matrix_dimension)
    if len(args) > 1:
        matrix_power = int(args[1])
        print("Provided power:", matrix_power)
    if len(args) > 2:
        matrix_type = 'Identity' if int(args[2]) == 0 else 'Random'
        print("Provided type:", matrix_type)
    if len(args) > 3:
        verbose = int(args[3]) == 1
        print("Provided verbose:", "Detailed" if verbose else "Quiet")
    print()
    
    if not (1 <= matrix_dimension <= 1000):
        print("Matrix dimension outside of acceptable range. Please enter a number in the range: [1, 1000]")
        sys.exit(1)
    if not (0 <= matrix_power <= 10000):
        print("Matrix power outside of acceptable range. Please enter a number in the range: [0, 10000]")
        sys.exit(2)
    
    benchmark(matrix_dimension, matrix_power, matrix_type, verbose)
    """
    for power in [600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
        benchmark(50, power, 'Random', False, "power")
        
    for dimension in [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]:
        benchmark(dimension, 1000, 'Random', False, "dimension")
        
    
