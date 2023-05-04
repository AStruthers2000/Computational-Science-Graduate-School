# -*- coding: utf-8 -*-
"""
Created on Mon May  1 20:36:30 2023

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

import numpy as np
import math
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# Define the function to be approximated
def f(x, y):
    return math.sin((math.pi * 10 * x) + (10 / (1 + y**2))) + math.log(x**2 + y**2)

# Generate the training and testing data
n_train, n_test = 5000, 2500

input_train = np.random.uniform(1, 100, size=(n_train, 2))
calculated_output_train = np.array([f(x[0], x[1]) for x in input_train])

input_test = np.random.uniform(1, 100, size=(n_test, 2))
calculated_output_test = np.array([f(x[0], x[1]) for x in input_test])

# Specify the MLP architecture
# n_hidden_layers = int(input("Enter the number of hidden layers: "))
# n_neurons = int(input("Enter the number of neurons per hidden layer: "))
print(f"There are {len(input_train)} (x, y) pairs and outputs used to train, with {len(input_test)} test case (x, y) pairs\n\n")
min_hidden_layers = -1
min_neurons = -1
min_rmse = 1000

for n_hidden_layers in [1, 2]:
    for n_neurons in range(1, 20, 1):
        mlp = MLPRegressor(
                    hidden_layer_sizes=(n_neurons,) * n_hidden_layers, 
                    activation='logistic',
                    solver='adam', 
                    learning_rate='constant',
                    learning_rate_init=0.001,
                    max_iter=1000000,
                    random_state=10
            )

        # Train the MLP on the training data
        mlp.fit(input_train, calculated_output_train)
        
        # Evaluate the MLP on the testing data
        predicted_output = mlp.predict(input_test)
        mse = mean_squared_error(calculated_output_test, predicted_output)
        rmse = np.sqrt(mse)
        print(f"Root-mean-squared error on test data with {n_hidden_layers} layers and {'0' if n_neurons < 10 else ''}{n_neurons} neurons per layer: {rmse}")
        if rmse < min_rmse:
            min_rmse = rmse
            min_hidden_layers = n_hidden_layers
            min_neurons = n_neurons

print(f"\nThe result with the smallest rmse of {min_rmse} was given by {min_hidden_layers} layer(s) with {min_neurons} neuron(s) each")
