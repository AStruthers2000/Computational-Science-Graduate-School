# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 15:44:56 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

import math
from tabulate import tabulate

def S(t, n, T):
    tot = 0
    
    #calculating the summation in equation 3.18
    for i in range(1, n+1):
        tot += (1/(2*i-1))*math.sin((2*(2*i-1)*math.pi*t)/T)
    tot *= 4/math.pi
    
    return tot

def f(t, T):
    val = 0
    
    #calculating the piecewise function at a given t value as in equation 3.17
    if 0 < t and t < T/2:
        val = 1
    elif t == T/2:
        val = 0
    else:
        val = -1
    return val


#setting up given n and alpha values
cases = [1,3,5,10,30,100]
alpha = [0.01, 0.25,0.49]
T = 2*math.pi

data = []

for n in cases:
    for a in alpha:
        t = a * T
        
        f_est = f(t, T)
        s_est = S(t, n, T)
        
        abs_err = abs(f_est - s_est)
        rel_err = abs(abs_err / f_est)
        
        #appending results to a 2D list for the tabulate function
        data.append([n, a, t, f_est, s_est, abs_err, rel_err])
        
print(tabulate(data, headers=["n", "alpha", "t", "f(t)", "S(t)", "Absolute Error", "Percent Error"]))

"""
Looking at the results table as well as a plot of the function on Desmos, it appears
as though the Fourier approximation of the function is most accurate towards the 
mid-point between 0 and T/2, and the mid-point between T/2 and T. Using the given t values,
we know that 2pi(0.01) and 2pi(0.49) are very close to the points where the piecewise function
goes from 1 to -1, which is where the Fourier approximation is least accurate. Therefore, it makes
sense that when alpha=0.25 we have the most accurate approximation, regardless of the value of n. 
The quality of the approximation, therefore, relies on how close alpha is to 0.25 or 0.75.

Regardless of the alpha value, increasing n results in a more accurate approximation. The higher the 
n value, the closer the Fourier approximation is to the actual function. 

This means that the accuracy of the approximation improves as n -> infinity and as 
alpha -> 0.25 or alpha -> 0.75
"""
        