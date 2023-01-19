# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 09:55:13 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""

seconds = 1e09
minutes = seconds / 60
hours = minutes / 60
days = hours / 24
years = days / 365

#an alternate calculation is dividing the seconds by 3.154*10^7
years_alt = seconds / 3.154e07


print("One billion seconds is {0} years.".format(years))

#average life span in Norway (2019) is 82.91 years
print("A baby could reasonably expect to live this long" if (years >= 0 and years <= 82.91) else "A baby probably wouldn't reasonably expect to live this long")

calc_diff = abs(years - years_alt)
days_diff = calc_diff * 365
print("Calculating the years via two different methods resulted in a difference of {0} days".format(days_diff))
