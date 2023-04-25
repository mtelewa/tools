#!/usr/bin/env python

import sys
import numpy as np
from scipy import constants as sci

kcalpermol_to_Jpermol = 4184
A_to_m = 1e-10
Pa_to_MPa = 1e-6
m3_to_cm3 = 1e6
N_to_mN = 1e3
gpermol_to_kg = 1e-3/sci.N_A

# For LJ - argon fluid
epsilon = 0.234888 * kcalpermol_to_Jpermol / sci.N_A # J
sigma = 3.405 * A_to_m # m
mass = 39.948  # g/mol

if 'timestep' in sys.argv:
    timestep = np.float64(input('insert tau* value:')) * (sigma*np.sqrt(mass*gpermol_to_kg/epsilon)) * 1e15
    print(f'Time step is {timestep:.2f} fs')
if 'temp' in sys.argv:
    temp = np.float64(input('insert T* value:')) * epsilon / sci.k
    print(f'Temperature is {temp:.2f} K')
if 'pressure' in sys.argv:
    press = np.float64(input('insert P* value:')) * epsilon * Pa_to_MPa / sigma**3
    print(f'Pressure is {press:.2f} MPa')
if 'density' in sys.argv:
    den = np.float64(input('insert ρ* value:')) * (mass/sci.N_A) / (sigma**3*m3_to_cm3)
    print(f'Density is {den:.2f} g/cm^3')
if 'shear_rate' in sys.argv:
    shear_rate = np.float64(input('insert ˙γ* value:')) * np.sqrt(epsilon/(mass*gpermol_to_kg)) / sigma
    print(f'Shear rate is {shear_rate:e} s^-1')
if 'surf_tension' in sys.argv:
    gamma = np.float64(input('insert gamma* value:')) * epsilon * N_to_mN / sigma**2
    print(f'Surface tension is {gamma:.2f} mN/m')
