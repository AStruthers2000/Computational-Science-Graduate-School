# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:14:18 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#it is important to use numpy to compute the values of exp, sin, and pi because Python
#can't natively compute the entire array of x values at a time and will try to convert
#the numpy array to a single float value if we use the math library instead of numpy
def f(x, t):
    return np.exp(-(x-3*t)**2)*np.sin(3*np.pi*(x-t))

def init():
    lines[0].set_data([], [])
    return lines

def frame(args):
    frame_no, t, x, lines = args
    y = f(x, t)
    lines[0].set_data(x, y)
    return lines


#there will be no temp png files because this code uses 
#FuncAnimation from MatPlotLib instead of mmfpeg
   
#removing any generated gif files
for filename in glob.glob('wavepacket*.*'):
    print("Deleting: {0}".format(filename))
    os.remove(filename)

#initalizing parameters
#we know that f_max = 1 because the primary driver of this function is the sin wave
x = np.linspace(-6, 6, 1001)
t_vals = np.linspace(-1, 1, 61)
f_max = 1

#creating inital empty plot
fig = plt.figure()
plt.axis([x[0], x[-1], -f_max, f_max])
lines = plt.plot([], [])
plt.xlabel('x')
plt.ylabel('y')

#creating arguments for every t value
"""
created object will be a list of tuples with the following form:
[(0, -1, [-6, ..., 6], [Line2D]), (1, -0.96666, [-6, ..., 6], [Line2D]), ...]

The objects in the tuple are:
0. the index of the current t value
1. the subsequent t value
2. the entire array of x values to plot
3. the reference to the plot lines
"""
all_args = [(frame_no, t, x, lines) for frame_no, t in enumerate(t_vals)]


anim = animation.FuncAnimation(fig, frame, all_args, interval=150, init_func=init, blit=True)

anim.save('wavepacket.gif', fps=9001) #over 9000? yes ✓
plt.show()

