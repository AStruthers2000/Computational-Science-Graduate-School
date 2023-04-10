# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 16:33:20 2023

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

x1 = [2, 1, -1]
x2 = [0, -1, -1]

w = [0, 1, 0]

t1 = -1
t2 = 1
n = 1

cases_passed = 0

while cases_passed < 2:
    net1 = sum([w[i] * x1[i] for i in range(len(w))])
    act = 1 if net1 > 0 else -1
    if act != t1:
        print("correcting weights")
        print(n*(t1-net1)*x1)
        print(n*(t1-net1))
        error = [n*(t1-net1)*x1[i] for i in range(len(x1))]
        w = [w[i] + error[i] for i in range(len(error))]
    else:
        cases_passed += 1
        
    print(w)
    break