# -*- coding: utf-8 -*-
"""
Created on Fri Dec  2 16:43:24 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""


import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from timeit import default_timer as timer


class Rectangle:
    #a pretty basic rectangle class that contains some helpful functions for checking if points are inside or outside of the rectangle
    
    def __init__(self, xl, xh, yl, yh):
        self.xl = xl
        self.yl = yl
        self.xh = xh
        self.yh = yh
    
    def GetXBounds(self):
        return [self.xl, self.xh]
    
    def GetYBounds(self):
        return [self.yl, self.yh]
    
    def IsPointInRect(self, point):
        inXMin = numpy.all(point[0] > self.xl)
        inXMax = numpy.all(point[0] < self.xh)
        inYMin = numpy.all(point[1] > self.yl)
        inYMax = numpy.all(point[1] < self.yh)
        return inXMin and inXMax and inYMin and inYMax
    
    def OutsideXMin(self, x):
        return x <= self.xl
    
    def OutsideXMax(self, x):
        return x >= self.xh
    
    def OutsideYMin(self, y):
        return y <= self.yl
    
    def OutsideYMax(self, y):
        return y >= self.yh


def init():
    #set the axis scale to just slightly larger than the box we are simulating
    plt.axis([rect.GetXBounds()[0] - 0.1, rect.GetXBounds()[1] + 0.1, rect.GetYBounds()[0] - 0.1, rect.GetYBounds()[1] + 0.1])
    
    #For FuncAnimation to work, we need to return a sequence of Matplotlib Artists
    #in our case, we only have one Artist, so we add the trailing comma and no second Artist to trick FuncAnimation
    return s_plt,


def frame(step):
    #tell python to use the xpositions and ypositions globally, instead of treating them like new local variables each frame
    global xpos
    global ypos

    #print(f"Calculating step {step+1}")
    
    #calculate the next position using the vectorized method for speed
    xpos, ypos = random_walk_2D(xpos, ypos, moves[step])
    
    #check to make sure that no particles randomly decided they wanted out of the simulation
    assert len(xpos) == np
    assert len(ypos) == np
    
    #set_offsets expects a Nx2 ndarray but we have two 1d position arrays
    #we can stack them together and create an ndarray with numpy.hstack
    #then give the scatter plot this frame's data
    data = numpy.hstack((xpos[:,numpy.newaxis], ypos[:,numpy.newaxis]))
    s_plt.set_offsets(data)
    
    #we update the title to reflect the current step (with a stupid conditional so that frame 1 is singular)
    plt.title(f"{np} particles after {step+1} step"+("s" if step+1>1 else ""))
    
    #again, we need to return a sequence of Artists to trick FuncAnimation into thinking we are giving it what it wants
    return s_plt,


def random_walk_2D(xpos, ypos, this_move):
    #vectorized calculation of the random walk algorithms, based off of the current frame's list of movements
    #if the particle is told to move north, we want to move the y position up by particle_step, for example
    ypos += numpy.where(this_move == NORTH, particle_step, 0)
    ypos -= numpy.where(this_move == SOUTH, particle_step, 0)
    xpos += numpy.where(this_move == EAST,  particle_step, 0)
    xpos -= numpy.where(this_move == WEST,  particle_step, 0)
    
    #after we move the particles, we need to validate that they are still within the box we care about
    #if a particle is outside of the box, we do the exact opposite of the move that resulted in the particle momentarily escaping
    ypos -= numpy.where(rect.OutsideYMax(ypos), particle_step, 0)
    ypos += numpy.where(rect.OutsideYMin(ypos), particle_step, 0)
    xpos -= numpy.where(rect.OutsideXMax(xpos), particle_step, 0)
    xpos += numpy.where(rect.OutsideXMin(xpos), particle_step, 0)
    
    #since I wasn't confident in the opposite movement code above, we want to validate that all particles are still inside the box
    #we do this by checking if each point is inside the rectangle. If it is, we add a 1 to the proper index of valid_points
    #and a 0 if not. Then, we assert that all elements in valid_points is equal to 1. If any element is 0, the assert will fail
    #and we will know that the opposite movement code failed somewhere
    valid_points = numpy.where(rect.IsPointInRect([xpos, ypos]), 1, 0)
    assert numpy.all(valid_points==1)

    return xpos, ypos


#defining the possible movement directions
NORTH = 1
SOUTH = 2
WEST = 3
EAST = 4

#creating the box that we are simulating
rect = Rectangle(0, 1, 0, 1)

#creating 10,000 points and simulating for 250 steps
#this number of steps has been validated to be a sufficient number of steps to achieve a fully mixed effect
np = 10000
ns = 400

#the distance, per step, that a particle can move
particle_step = 0.05

#setting up a vectorized version of every move that each particle will make, and shaping the data into 
#a matrix with ns rows and np columns
moves = numpy.random.randint(1, 4+1, size = ns*np)
moves.shape = (ns, np)

#setting up the initial uniformly distributed particles inside a box of dimensions x=[0, 1/2] and y = [0, 1]
xpos = numpy.random.uniform(rect.GetXBounds()[0], rect.GetXBounds()[1]/2.0, np)
ypos = numpy.random.uniform(rect.GetYBounds()[0], rect.GetXBounds()[1], np)

fig = plt.figure()

#drawing the actual boundries of the box we are simulating for clarity and nice visuals
plt.hlines(rect.GetYBounds()[0], rect.GetXBounds()[0], rect.GetXBounds()[1], colors='black')
plt.hlines(rect.GetYBounds()[1], rect.GetXBounds()[0], rect.GetXBounds()[1], colors='black')
plt.vlines(rect.GetXBounds()[0], rect.GetYBounds()[0], rect.GetYBounds()[1], colors='black')
plt.vlines(rect.GetXBounds()[1], rect.GetYBounds()[0], rect.GetYBounds()[1], colors='black')

#initialize the scatter plot with a blank plot and dot size of 3
s_plt = plt.scatter([], [], s = 3)

#generating a list of all frame numbers to pass to FuncAnimation, so that we don't have to keep track of the current frame number
steps = [frame_no for frame_no in range(ns)]

start_anim_time = timer()

#setting up and starting the animation
anim = animation.FuncAnimation(fig, frame, steps, interval = 1, init_func = init, blit = True)
anim.save('GasMolecules.gif', fps=144)

finish_anim_time = timer()
exe_time = finish_anim_time - start_anim_time
print(f"Entire animation took {exe_time} seconds to generate\n\tThat's an average of {exe_time/ns} seconds per frame!")