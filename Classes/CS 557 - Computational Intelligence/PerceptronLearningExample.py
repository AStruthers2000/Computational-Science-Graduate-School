# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:33:20 2023

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

def dot(x, y):
    assert len(x) == len(y)
    return sum([x[i] * y[i] for i in range(len(w))])

def scalar_multiply(a, s):
    return [x * s for x in a]

inputs = [[2, 1, -1],
          [0, -1, -1]]

w = [0, 1, 0]

t = [-1, 1]

o = [999 for i in range(len(t))]

n = 1

cases_passed = 0

while t != o:
    for i in range(len(inputs)):
        x = inputs[i]
        net = dot(w, x)

        act = 1 if net > 0 else -1
        if act != t[i]:
            print("correcting weights")
            error = scalar_multiply(x, n*(t[i] - net))
            w = [w[i] + error[i] for i in range(len(error))]
        else:
            o[i] = act
        print(w)
