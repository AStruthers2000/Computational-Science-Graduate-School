# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 14:09:59 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

a = 1/947.0*947
b = 1
tol = 0.000001

if a != b:
    print("Wrong result!")

if abs(a-b)<tol:
    print("Correct result!")
    