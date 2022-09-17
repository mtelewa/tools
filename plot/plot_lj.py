import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('imtek')

def lj(r, sigma, epsilon, wca=None):
    """
    r, array of inter particle separation
    """

    u = 4*epsilon*((sigma/r)**12-(sigma/r)**6)

    if wca is not None:
        for i in range(len(r)):
            if r[i]>(2**(1/6)*sigma):
                u_wca[i]=0

    return u


if __name__ =' __main__':

    r = np.linspace(2.8, 7, 100)
    u = lj(r, 3.405, 0.23488) # For argon, units: Angstrom and Kcal/mol

    if 'wca' in sys.argv:
        u = lj(r, 3.405, 0.23488, wca=1) # For argon, units: Angstrom and Kcal/mol

    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax.set_xlabel('$r_{ij}$')
    ax.set_ylabel('$U(r_{ij})$')
    ax.axhline(y = 0, ls='--', color='k')
    ax.legend(frameon=False)

    ax.plot(r, u, label='Lennard-Jones')

    if 'wca' in sys.argv:
        ax.plot(r, u_wca, label='WCA')

    fig.savefig('lj.png' , format='png')
