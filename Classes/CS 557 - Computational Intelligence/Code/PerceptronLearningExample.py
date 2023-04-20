# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:33:20 2023

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

from random import randint
import math

def dot(x, y):
    assert len(x) == len(y)
    return sum([x[i] * y[i] for i in range(len(w))])

def scalar_multiply(a, s):
    return [x * s for x in a]

inputs = [[2, 1, 0, 4, 1, 39, 2, -1],
          [0, -1, 8, 9, -1, 2, 10, -1],
          [1, -1, 4, 5, 3, 2, -2, -1],
          [0, -1, 3, 4, -1, 2, 1, -1]]

w = [randint(-1, 1) for _ in range(len(inputs[0]))]

t = [-1, 1, 1, -1]

o = [999 for i in range(len(t))]

n = 0.01

cases_passed = 0

while t != o:
    print(f"Current weights: {w}")
    print(f"Current outputs: {o}")
    print(f"Desired outputs: {t}")
    print("=========="*5)
    for i in range(len(inputs)):
        x = inputs[i]
        net = dot(w, x)

        act = 1 if net > 0 else -1
        if act != t[i]:
            error = scalar_multiply(x, n*(t[i] - net))
            w = [w[i] + error[i] for i in range(len(error))]
        o[i] = act
    if sum([1 if math.isnan(w[i]) else 0 for i in range(len(w))]) > 0:
        print("Detected NaN")
        break

print("Final weights:   {0}".format(w))
print("Final outputs:   {0}".format(o))
print("Desired outputs: {0}".format(t))
print("=========="*5)
        
