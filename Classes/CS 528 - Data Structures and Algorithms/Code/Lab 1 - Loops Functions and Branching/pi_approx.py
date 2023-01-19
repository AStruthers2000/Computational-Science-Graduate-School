# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:16:35 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

import math

def pathlength(x, y):
    n = len(x)
    l = 0
    
    #assert will fail if the inputs to the function are different lengths
    #which signifies faulty input
    assert n == len(y)
    
    #computing the summation from figure 3.16
    for i in range(1, n):
        l += math.sqrt((x[i] - x[i-1])**2 + (y[i]-y[i-1])**2)
        
    return l


#Creating list of k values
test_vals = [_ for _ in range(2, 10+1)]

for k in test_vals:
    n = 2**k
    
    x = []
    y = []
    
    for i in range(n+1):
        #calculating xi and yi with the given formulas
        xi = 0.5*math.cos((2*math.pi*i)/n)
        yi = 0.5*math.sin((2*math.pi*i)/n)
        
        x.append(xi)
        y.append(yi)
    
    pi_est = pathlength(x, y)
    abs_err = abs(pi_est - math.pi)
    rel_err = abs_err / math.pi
    
    print("With n = {4} points:\n\tPI Estimate: \t\t{0}\n\tPI Actual: \t\t\t{1}\n\tAbsolute Error: \t{2}\n\tRelative Error: \t{3}".format(pi_est, math.pi, abs_err, rel_err, n))

    print("====="*10)    