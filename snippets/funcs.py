#!/usr/bin/env python
# -*- coding: utf-8 -*-

def quadratic(x,a,b,c):
    return a*x**2+b*x+c

# Define parabola derivative
def quad_slope(x,a,b):
    return 2*a*x+b


def linear(x,a,b):
    return a*x+b

def step(x):
    y = []
    for i in x:
        if 0 < i <= 0.2*np.max(length):
            y.append(1)
        else:
            y.append(0)
    return y

def quartic(x):
    y = []
    for idx, val in enumerate(x):
        if 0 <= val <= (smoothed_pump_length):
            xu = (val / ((pump_length*15/8)/2) ) - 1
            val2 = (xu**4) - (2*xu**2) + 1
            y.append(val2)
        else:
            y.append(0)
    return y
