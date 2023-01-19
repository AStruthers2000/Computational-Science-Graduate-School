# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 13:59:21 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

import numpy as np
MALE = 1
FEMALE = 2

def advance_generation(parents, male_portion, fertility, violation_rate, desired_children=4):
    males = len(parents[parents==MALE])
    females = len(parents[parents==FEMALE])
    assert males+females == len(parents)
    
    couples = min(males, females)
    
    children = get_children(couples, male_portion, fertility)
    max_children = 1
    
    daughters = children[children==FEMALE]
    
    while len(daughters)>0:
        new_children = get_children(len(daughters), male_portion, fertility)
        children = np.concatenate((children, new_children))
        daughters = new_children[new_children==FEMALE]
        max_children+=1
        
    law_violations = get_children(int(len(children)*violation_rate)*desired_children, male_portion, fertility=1.0)
    children = np.concatenate((children, law_violations))
    
    return children, max_children


def get_children(n, male_portion, fertility):
    if n==0: return []
    n = int(fertility*n)
    r = np.random.random(n)
    children = np.zeros(n, int)
    
    children[r<male_portion] = MALE
    children[r>=male_portion] = FEMALE
    return children



N = 1000000

male_portion = 0.51
fertility = 0.92
violation_rate = 0.06

desired_children = 6

gen = 10

parents = get_children(N, male_portion=0.5, fertility=1.0)

print("One son policy starts with {0} parents".format(len(parents)))

prev_x = N
r_list = []
for i in range(gen):
    parents, mc = advance_generation(parents, male_portion, fertility, violation_rate, desired_children)    
    print("Generation {0}: {1} (max children in a family: {2})".format(i+1, len(parents), mc))
    
    if prev_x == -1:
        prev_x = len(parents)
        continue
    else:
        x = len(parents)
        r = (x/prev_x) - 1
        r_list.append(r)
        prev_x = x
        print("The rate of growth between last generation and current: {0}".format(r))
        
print("The average rate of growth for 10 generations is: {0}".format(sum(r_list)/len(r_list)))
        
"""
Looking at the output rate of growth, we can assume that the growth equation
would have approximately r = 0.116

x_n = (1+0.116)x_{n-1}

From this analysis we can see that the population increases by ~11.6% per generation
"""