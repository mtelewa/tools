import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.style.use('thesis-3sub')
# Specify the path to the font file
font_path = '/usr/share/fonts/truetype/LinLibertine_Rah.ttf'
# Register the font with Matplotlib
fm.fontManager.addfont(font_path)
# Set the font family for all text elements
plt.rcParams['font.family'] = 'Linux Libertine'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Linux Libertine'
plt.rcParams['mathtext.it'] = 'Linux Libertine:italic'
plt.rcParams['mathtext.bf'] = 'Linux Libertine:bold'

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
    u_intermed = lj(r, 2.655, 0.1618)#0.0809) #(r, 3.405, 0.23488) # For argon, units: Angstrom and Kcal/mol
    u_slip = lj(r, 2.655, 0.00809)
    #if 'wca' in sys.argv: u_wca = lj(r, 3.405, 0.23488, wca=1) # For argon, units: Angstrom and Kcal/mol
    
    fig, ax = plt.subplots(nrows=1, ncols=1, sharex=True) #figsize=(8.6,6.4))
    #plt.subplots_adjust(left=0.2, right=0.9, bottom=0.15, top=0.9)

    ax.set_xlabel(r'Separation $r_{ij} \, \mathrm{(\AA)}$')
    ax.set_ylabel('Energy $U(r_{ij})$ (Kcal/mol)')
    ax.axhline(y = 0, ls='--', color='k')

    ax.plot(r, u_stick, label='Wetting')
    ax.plot(r, u_intermed, label='Intermediate')
    ax.plot(r, u_slip, label='Non-wetting')

    ax.set_ylim(top=0.6)
    ax.legend(frameon=False, loc='lower right')
    #if 'wca' in sys.argv: ax.plot(r, u_wca, label='WCA')
    fig.savefig('lj.png' , format='png')
