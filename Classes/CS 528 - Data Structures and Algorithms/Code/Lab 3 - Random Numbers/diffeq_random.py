# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 10:12:15 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""
import matplotlib.pyplot as plt


def calc_next(n):
    #grab the previous number in a more readable way
    prev_num = x_seq[-1]
    
    #calculate and add the new number to the sequence
    x_seq.append((a*prev_num+c)%m)
        

def random_number(n, x_0):    
    #used to start the sequence
    if len(x_seq)==0:
        x_seq.append((a*x_0+c)%m)
        
    #calculate the next value in the sequence
    calc_next(n)
    
    #divide by m
    y = x_seq[n]/m
    return y

    
def number_generator(seed, N=1):
    nums = []
    for i in range(N):    
        nums.append(random_number(i, seed))
    x_seq.clear()
    return nums

#initial parameters
a = 8121
c = 28411
m = 134456

x_seq = []

#seed can be set to anything. i just button mashed the numpad
seed = 1301232

#I found N>=50,000 provides enough samples to see the uniformity
#N=1,000,000 makes a histogram that is almost perfectly flat
N=100000
sequence = number_generator(seed, N)

#generating a histogram with 50 bins
n, bins, patches = plt.hist(sequence, 50, density=True, facecolor='b', alpha=0.75)
plt.xlabel("Distribution")
plt.ylabel("Value")
plt.title("Histogram of {0} Random Number Generation".format(N))
plt.savefig("RandomDistribution-{0}.png".format(N))
plt.show()
