# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 14:58:45 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

import glob, os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

#removing any generated files
for filename in glob.glob('pipeflow*.*'):
    print("Deleting: {0}".format(filename))
    os.remove(filename)

B = 0.02
u = 0.02
n = 0.1
R = 1
norm = True

def v(r, n):
    val = ((B/(2*u))**(1/n)) * (n/(n+1)) * (R**(1+(1/n))-r**(1+(1/n)))    
    return val

"""
plotting the inital graph for part b
"""
r = np.linspace(0, R, 1001)
y = v(r, n)

plt.xlabel('x')
plt.ylabel('y')
plt.title("n={0}, B={1}, u={2}, R={3}".format(round(n, 3), B, u, R))
plt.plot(r, y)
plt.savefig("pipeflow_part_b.png")
plt.show()


"""
creating the normed animation for part c
"""

#initalizing parameter
n_vals = np.linspace(1, 0.01, 61)

#creating inital empty plot
fig = plt.figure()
plt.axis([r[0], r[-1], 0, 1.2])
lines = plt.plot([], [])
plt.xlabel('x')
plt.ylabel('y')

def init():
    lines[0].set_data([], [])
    return lines

def frame(args):
    frame_no, n, r, lines = args
    y = v(r, n)
    if norm:
        y /= v(0, n)
    plt.title("n={0}, B={1}, u={2}, R={3}".format(round(n, 3), B, u, R))
    lines[0].set_data(r, y)
    return lines

#creating arguments for every n value
all_args = [(frame_no, n, r, lines) for frame_no, n in enumerate(n_vals)]

anim = animation.FuncAnimation(fig, frame, all_args, interval=150, init_func=init, blit=True)

anim.save('pipeflow.gif', fps=144)
plt.show()

