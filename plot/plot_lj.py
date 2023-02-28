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


if __name__ == '__main__':
    r = np.linspace(2.4, 7, 100)
    
    u_stick = lj(r, 2.655, 0.809)
    u_intermed = lj(r, 2.655, 0.0809) #(r, 3.405, 0.23488) # For argon, units: Angstrom and Kcal/mol
    u_slip = lj(r, 2.655, 0.00809)

    #if 'wca' in sys.argv: u_wca = lj(r, 3.405, 0.23488, wca=1) # For argon, units: Angstrom and Kcal/mol
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True)
    ax.set_xlabel('Separation $r_{ij} (\AA)$')
    ax.set_ylabel('Energy $U(r_{ij})$ (Kcal/mol)')
    ax.axhline(y = 0, ls='--', color='k')

    ax.plot(r, u_stick, label='Wetting')
    ax.plot(r, u_intermed, label='Intermediate wetting')
    ax.plot(r, u_slip, label='Non-wetting')

    ax.set_ylim(top=0.6)

    ax.legend(frameon=False)

    #if 'wca' in sys.argv: ax.plot(r, u_wca, label='WCA')

    fig.savefig('lj.png' , format='png')
