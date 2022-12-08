#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from scipy.optimize import curve_fit


def quadratic(x,a,b,c):
    return a*x**2+b*x+c


# Parabola derivative
def quad_slope(x,a,b):
    return np.abs(2*a*x+b)


def linear(x,a,b):
    x = np.array(x)
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


def power(x,a,b,c):
    return a*np.power(x,b) + c

def power_new(x,a,b,c):
    return a*np.power(1-(x/b),c)

def density_scaling(x,a,c):
    # 3.125 is 1/beta where beta is the critical exponent (Martin and Siepmann J. Phys. Chem. B, Vol. 102, No. 14, 1998)
    return a*np.power(x,3.125) + c

def rectilinear_diameters(x,a,b):
    # 147.2 is the critical temperature of argon from my simulations
    # 465.857 is the critical temperature of n-pentane from my simulations
    return a*x-b+147.2

def solve_quadratic(a,b,c):
    d = np.abs(b**2 - (4*a*c))
    sol1 = (-b - np.sqrt(d)) / (2*a)
    sol2 = (-b + np.sqrt(d)) / (2*a)
    return sol1, sol2

def quadratic_cosine_series(x,a,b,c,d,e,f,g,h,i,j,k,l,m):

    height=x[-1]+x[0]

    y = a*x**2 + b*x + c +\
          (d *np.cos(2*np.pi*x/height)) + (e *np.cos(4*np.pi*x/height)) + \
          (f *np.cos(6*np.pi*x/height)) + (g *np.cos(8*np.pi*x/height)) + \
          (h *np.cos(10*np.pi*x/height)) + (i *np.cos(12*np.pi*x/height)) + \
          (j *np.cos(14*np.pi*x/height)) + (k *np.cos(16*np.pi*x/height)) + \
          (l *np.cos(18*np.pi*x/height)) + (m *np.cos(20*np.pi*x/height))

    return y


def quartic_cosine_series(x,a,b,c,d,e,f,g,h,i,j,k,l,m,y,z):

    height=x[-1]+x[0]

    y = a*x**4 + b*x**3 + c*x**2 + z*x + y +\
          (d *np.cos(2*np.pi*x/height)) + (e *np.cos(4*np.pi*x/height)) + \
          (f *np.cos(6*np.pi*x/height)) + (g *np.cos(8*np.pi*x/height)) + \
          (h *np.cos(10*np.pi*x/height)) + (i *np.cos(12*np.pi*x/height)) + \
          (j *np.cos(14*np.pi*x/height)) + (k *np.cos(16*np.pi*x/height)) + \
          (l *np.cos(18*np.pi*x/height)) + (m *np.cos(20*np.pi*x/height))

    return y


def fit(x,y,order):
    """
    Returns
    -------
    xdata : arr
        x-axis range for the fitting parabola
    polynom(xdata): floats
        Fitting parameters for the velocity profile
    """
    # Fitting coefficients
    coeffs_fit = np.polyfit(x, y, order)     #returns the polynomial coefficients
    # construct the polynomial
    polynom = np.poly1d(coeffs_fit)

    return {'fit_data': polynom(x), 'coeffs':coeffs_fit}


def fit_with_cos(x,y,order):

    if order == 2:
        coeffs,_ = curve_fit(quadratic_cosine_series, x, y)
        fitted = funcs.quadratic_cosine_series(x,*coeffs)
    if order == 4:
        coeffs,_ = curve_fit(quartic_cosine_series, x, y)
        fitted = funcs.quartic_cosine_series(x,*coeffs)

    return {'fit_data': fitted, 'coeffs':coeffs}


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
        return y.real, -y[1:-1].imag


def fourierSeries(coeffs,x,n):
    """This functions returns the value of the Fourier series for a given value
    of x given the already calculated Fourier coefficients.
    """
    value = fourier_series_coeff(1,n)[0]
    l=x[-1]
    for i in range(1,n+1):
        value = value + coeffs[1][i-1]*np.cos(i*np.pi*x/l) #+  coeffs[2][i-1]*np.sin(i*np.pi*x/l)

    return value


def fit_with_fourier(x,y,order):
    T = 1
    n_coeffs = len(xdata)
    coeffs = fourier_series_coeff(np.cos, T, n_coeffs)[0]
    for i in range(1,n+1):
        fit_data.append(dd.fit(x,y,order)['fit_data'] + coeffs[1][i-1]*np.cos(i*np.pi*x/l))

    return {'fit_data': fitted, 'coeffs':coeffs}


def RP(y, t, rho, pv, pl):
    """ Rayleigh-Plesset equation assuming viscosity = surface tension = 0
    """
    R, Rdot = y
    dydt = [Rdot, -3/(2*R)*Rdot**2 + ((pv-pl)/(rho*R))]
    return dydt


def RP_full(y, t, rho, pv, pl, gamma, eta):
    """ Rayleigh-Plesset equation with non-zero eta and gamma
    """
    R, Rdot = y
    dydt = [Rdot, -3/(2*R)*Rdot**2 + ((pv-pl)/(rho*R)) - (2*gamma/R) - (4*eta*Rdot/R)]
    return dydt
