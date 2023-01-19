# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 15:08:52 2022

@author: Andrew Struthers
@honor-code: I pledge that I have neither given nor received help from anyone 
             other than the instructor or the TAs for all program components 
             included here.
"""
import math
import matplotlib.pyplot as plt

"""
D_t = ap_k + b
S_t = Ap_{t-1} + B + ln(1 + p_{t-1})

For part a), we must find a p-value such that D_t=S_t, so we have
ap + b = Ap + B + ln(1 + p)

We know that A=1, a=-3, B=0, and b=5
-3p + 5 = p + ln(1 + p)
-4p + 5 = ln(1 + p)

Putting this into an equation solver, we have
p = 1.06831627888686...

Now for part b)
"""

A=1
a=-3
B=0
b=5
p = 4.5

def calc_p(p, t):
    """
    we know that D_t=S_t, and we need to solve for p_t. so we have:
    ap_t + b = Ap_{t-1} + B + ln(1 + p_{t-1})
    ap_t = Ap_{t-1} + B + ln(1 + p_{t-1}) - b
    p_t = (Ap_{t-1} + B + ln(1 + p_{t-1}) - b) / a
    """
    prev_p = p
    p = (A*prev_p+B+math.log(1+prev_p)-b)/a 
    return prev_p, p

def calc_s(p, t):
    #i am passing the previous p-value to this function, so the calculation just relies on variable p
    s = A*p+B+math.log(1+p)
    return s

#we will calculate N points, but only graph the first plot_points years of data
#(since nothing interesting happens after basically t=10, but we still want the long term trend)
N = 1000000
plot_points = 50

p_vals = [p]
s_vals = []
for t in range(1, N):
    prev_p, p = calc_p(p, t)
    
    #calculate s_t using the previous p-value
    #it should be the current p-value, but since we updated p in the previous line
    #the previous p-value is technically the current year's price
    s = calc_s(prev_p, t)
    
    if len(p_vals) < 2*plot_points:
        p_vals.append(p)
        s_vals.append(s)

#only plot the first plot_points p and s values
plt.plot(p_vals[:plot_points])

#have to offset s_vals since there is no calculatable initial supply when only given one price value
plt.plot([x for x in range(1, len(s_vals[:plot_points])+1)], s_vals[:plot_points], linestyle='dashed')

#set a nice title and axis labels
plt.title("Supply and Price of Wheat")
plt.xlabel("Time (Years)")
plt.ylabel("Wheat Value (Dollars per Bushel)")
plt.legend(["Wheat Price", "Wheat Supply"])

#add an annotation with the long term price of wheat, rounded to 3 sig-figs
#side note: the full, unrounded price is 1.0683162788868632 USd/BU
#which just so happens to be the exact value that we calculated in part a.  
plt.annotate(f"Long term trend of the\nprice of wheat: {round(p_vals[-1],3)} USd/BU\nat t={N}", (25,2.5))

plt.savefig("Wheat.png")
plt.show()