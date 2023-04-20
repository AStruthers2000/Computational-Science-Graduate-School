import numpy as np
from timeit import default_timer as timer

def print_vector(v):
    for x in range(3):
        string = ""
        for y in range(3):
            i = x * 3 + y
            string += f"{v[i]} "
        print(string)

def stoa(s):
    v = [int(x) for x in s]
    v.append(-1)
    return np.array(v)

def train(training_set, w, n):
    weights = w
    learning_rate = n

    # Train the neuron
    for i in range(100000):
        # Select a random input vector and desired output from the training set
        input_vec, desired_output = training_set[np.random.randint(len(training_set))]

        # Compute the actual output of the neuron
        actual_output = np.sign(np.dot(weights, input_vec))

        # Update the weights if the actual output doesn't match the desired output
        if actual_output != desired_output:
            weights += learning_rate * (desired_output - actual_output) * input_vec

    return weights


def classify(test_set, test_set_out, weights, set_type, verbose = False):
    # Classify the test set and compute the classification accuracy
    num_correct = 0
    for i in range(len(test_set)):
        input_vec = test_set[i]
        desired_output = test_set_out[i]
        actual_output = np.sign(np.dot(weights, input_vec))

        if verbose:
            print("{0} Classified: ".format("Correctly" if desired_output==actual_output else "Incorrectly"))
            print_vector(input_vec)
            print("as: {0}".format("L" if actual_output == L_out else "I"))
            print("====="*5)
            
        if actual_output == desired_output:
            num_correct += 1

    accuracy = num_correct / len(test_set) * 100

    print("Classification accuracy for {} set: {:.2f}%".format(set_type, accuracy))


# Define the input vectors for L and I
L_data = [
    stoa("100100111"),
    stoa("100100110"),
    stoa("100111000"),
    stoa("100110000"),
    stoa("010010011"),
    stoa("010011000"),
    stoa("010010110"),
    stoa("001001111"),
    stoa("001001011")
]

I_data = [
    stoa("100100100"),
    stoa("100100000"),
    stoa("010010010"),
    stoa("010010000"),
    stoa("001001001"),
    stoa("001001000"),
    stoa("000100100"),
    stoa("000010010"),
    stoa("000001001"),
    stoa("100010001")
]

# Define the desired outputs for L and I
L_out = 1
I_out = -1

# Combine the input vectors and desired outputs into a training set
training_set = []
for v in L_data:
    training_set.append((v, L_out))

for v in I_data:
    training_set.append((v, I_out))

np.random.shuffle(training_set)

# Initialize the weights randomly
weights = np.random.uniform(low=-1.0, high=1.0, size=10)

# Train the weights with the provided training set of vectors
start = timer()
weights = train(training_set, weights, 0.05)
end = timer()
print(f"Training took: {end - start} seconds")


# Validate the training worked properly
classify([v for (v, c) in training_set], [c for (v, c) in training_set], weights, "training")

# Define a test set of input vectors
test_set = [
    stoa("100100110"),  #L with short tail
    stoa("100101100"),  #I with noise
    stoa("101100110"),  #L with short tail and noise
    stoa("101001001"),  #I with noise
    stoa("100100001"),  #short I with noise
    stoa("110010011"),  #L with noise
]

# Define the correct outputs for the test set
test_set_out = np.array([L_out, I_out, L_out, I_out, I_out, L_out, L_out, L_out, L_out, I_out, I_out, I_out])

classify(test_set, test_set_out, weights, "test", True)
