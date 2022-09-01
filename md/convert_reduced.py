import numpy as np
from scipy import constants as sci

kcalpermol_to_j = 4184
A_to_m = 1e-10

# For LJ - argon fluid
epsilon = 0.234888 * kcalpermol_to_j / sci.N_A # J
sigma = 3.405 * A_to_m # m

# surface tension
gamma = np.float64(input('insert gamma* value:')) * epsilon / sigma**2
temp = np.float64(input('insert T* value:')) * epsilon / sci.k

print(f'Surface tension is {gamma*1e3} mN/m at a temperature of {temp} K.')
