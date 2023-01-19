# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 16:42:29 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

import math
from tabulate import tabulate

def diff(f, x, h=1e-05):
    f_prime = (f(x+h)-f(x-h))/(2*h)
    return f_prime


def test_diff():
    f = lambda x: 2*x**2+6*x+4
    
    #the derivative of f is f'(x) = 4x+6
    #at x = 0, f'(0) = 4(0) + 6 = 6
    
    exact = 6
    approx = diff(f, 0)
    
    success = abs(exact - approx) < 1e-10
    msg = "Numerical differentiation approximation failed for quadratic equation (approximation for quadratic is precise)"
    assert success, msg
    
    
def application():
    test_diff()
    
    data = []
    
    functions = {"e^x" : [lambda x: math.exp(x), 0, 1], 
                 "e^(-2(x^2))" : [lambda x: math.exp(-2*(x**2)), 0, 0],
                 "cos(x)" : [lambda x: math.cos(x), 2*math.pi, 0],
                 "ln(x)" : [lambda x: math.log(x), 1, 1]
                }
    
    for key, value in functions.items():
        h = 0.01
        
        f = value[0]
        x = value[1]
        exact = value[2]
        
        approx = diff(f, x, h)
        abs_err = abs(exact - approx)
        
        data.append([key, x, exact, approx, abs_err])
    
    print(tabulate(data, headers=["f(x)", "x", "Exact", "Approximation", "Error"]))
        
application()
