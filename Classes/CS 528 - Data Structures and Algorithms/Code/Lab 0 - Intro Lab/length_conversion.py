# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 10:32:47 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

meters = float(input("Enter a number of meters you want converted: "))

cm = meters * 100.0
inches = cm / 2.54
feet = inches / 12
yards = feet / 3
miles = yards / 1760

string = "Converting {0} meters to:\n"
string += "\tCentimeters:    {1}\n"
string += "\tInches:         {2}\n"
string += "\tFeet:           {3}\n"
string += "\tYards:          {4}\n"
string += "\tMiles:          {5}\n"

print(string.format(meters, cm, inches, feet, yards, miles))

