import numpy as np

def print_vector(v):
    for x in range(3):
        string = ""
        for y in range(3):
            i = x * 3 + y
            string += f"{v[i]} "
        print(string)

# Define the input vectors for L and I
L_data = np.array([
    [1, 0, 0,
     1, 0, 0,
     1, 1, 1],
    [1, 1, 1, 1, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 1, 0, 0]
])

I_data = np.array([
    [0, 1, 0, 0, 1, 0, 0, 1, 0],
    [0, 0, 0, 1, 1, 1, 0, 0, 0],
    [0, 1, 0, 1, 1, 1, 0, 1, 0]
])

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
for (v, o) in training_set:
    print_vector(v)
    print("I" if o == I_out else "L")
    print("=====")

# Initialize the weights randomly
weights = np.random.randn(9)

# Define the learning rate
learning_rate = 0.1

# Train the neuron
for i in range(1000):
    # Select a random input vector and desired output from the training set
    input_vec, desired_output = training_set[np.random.randint(len(training_set))]

    # Compute the actual output of the neuron
    actual_output = np.sign(np.dot(weights, input_vec))

    # Update the weights if the actual output doesn't match the desired output
    if actual_output != desired_output:
        weights += learning_rate * (desired_output - actual_output) * input_vec

# Define a test set of input vectors
test_set = np.array([
    [1, 0, 0, 1, 0, 0, 1, 1, 0],  # L with one missing pixel
    [0, 1, 0, 0, 1, 0, 0, 1, 1],  # I with one extra pixel
    [1, 1, 0, 0, 1, 0, 0, 1, 0],  # L with two extra pixels
    [0, 0, 0, 1, 1, 1, 0, 1, 0],  # I with two missing pixels
])

# Define the correct outputs for the test set
test_set_out = np.array([1, -1, 1, -1])

# Classify the test set and compute the classification accuracy
num_correct = 0
for i in range(len(test_set)):
    input_vec = test_set[i]
    desired_output = test_set_out[i]
    actual_output = np.sign(np.dot(weights, input_vec))
    if actual_output == desired_output:
        num_correct += 1

accuracy = num_correct / len(test_set) * 100

print("Classification accuracy for test set: {:.2f}%".format(accuracy))
