import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('imtek')

fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)

r = np.linspace(2.8,7,100)
sigma = 3.405 # Ang.  (Argon)
epsilon = 0.23488 # Kcal/mol  (Argon)

u = 4*epsilon*((sigma/r)**12-(sigma/r)**6)
u_wca = 24*epsilon*((sigma/r)**12-(sigma/r)**6)

# for i in range(len(r)):
#     if r[i]>(2**(1/6)*sigma):
#         u_wca[i]=0

ax.set_xlabel('$r_{ij}$')
ax.set_ylabel('$U(r_{ij})$')
ax.plot(r, u, label='Lennard-Jones')
ax.plot(r, u_wca, label='WCA')
ax.axhline(y = 0, ls='--', color='k')

ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')
ax.legend(frameon=False)
fig.savefig('lj.png' , format='png')
