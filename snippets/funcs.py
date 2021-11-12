#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

def quadratic(x,a,b,c):
    return a*x**2+b*x+c

def quadratic_cosine_series(x,a,b,c,d):
    return a*x**2+b*x+c+(d*np.cos(2*np.pi*x/h))


# Define parabola derivative
def quad_slope(x,a,b):
    return np.abs(2*a*x+b)


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


def power(x,a):
    return x**a


def quadratic_cosine_series(x,a,b,c,d,e,f,g):

    y = a*x**2 + b*x + c + \
          (d *np.cos(np.pi*x/h))   + (e *np.cos(2*np.pi*x/h)) + \
          (f *np.cos(3*np.pi*x/h)) + (g *np.cos(4*np.pi*x/h))

    return y


def fourier_series_coeff(f, T, N, return_complex=False):
    """Calculates the first 2*N+1 Fourier series coeff. of a periodic function.

    Given a periodic, function f(t) with period T, this function returns the
    coefficients a0, {a1,a2,...},{b1,b2,...} such that:

    f(t) ~= a0/2+ sum_{k=1}^{N} ( a_k*cos(2*pi*k*t/T) + b_k*sin(2*pi*k*t/T) )

    If return_complex is set to True, it returns instead the coefficients
    {c0,c1,c2,...}
    such that:

    f(t) ~= sum_{k=-N}^{N} c_k * exp(i*2*pi*k*t/T)

    where we define c_{-n} = complex_conjugate(c_{n})

    Refer to wikipedia for the relation between the real-valued and complex
    valued coeffs at http://en.wikipedia.org/wiki/Fourier_series.

    Parameters
    ----------
    f : the periodic function, a callable like f(t)
    T : the period of the function f, so that f(0)==f(T)
    N_max : the function will return the first N_max + 1 Fourier coeff.

    Returns
    -------
    if return_complex == False, the function returns:

    a0 : float
    a,b : numpy float arrays describing respectively the cosine and sine coeff.

    if return_complex == True, the function returns:

    c : numpy 1-dimensional complex-valued array of size N+1

    """
    # From Shannon theoreom we must use a sampling freq. larger than the maximum
    # frequency you want to catch in the signal.
    f_sample = 2 * N
    # we also need to use an integer sampling frequency, or the
    # points will not be equispaced between 0 and 1. We then add +2 to f_sample
    t, dt = np.linspace(0, T, f_sample + 2, endpoint=False, retstep=True)

    y = np.fft.rfft(f(t)) / t.size

    if return_complex:
        return y
    else:
        y *= 2
        return y[0].real, y[1:-1].real, -y[1:-1].imag
